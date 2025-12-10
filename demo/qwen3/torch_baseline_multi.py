import torch
import time
import argparse
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings('ignore')

class LLMBenchmark:
    def __init__(self, model_name, batch_size, device='cuda', compile_model=True, 
                 num_gpus=1):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.compile_model = compile_model
        self.num_gpus = num_gpus
        
        print(f"\n{'='*60}")
        print(f"Initializing: {model_name}")
        print(f"Batch Size: {batch_size}")
        print(f"Number of GPUs: {num_gpus}")
        if num_gpus > 1:
            for i in range(num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Attention Backend: FlashInfer")
        print(f"{'='*60}\n")
        
        # Check FlashInfer availability
        try:
            import flashinfer
            print("✓ FlashInfer detected")
        except ImportError:
            print("⚠ WARNING: FlashInfer not found!")
            print("  Install with: pip install flashinfer -f https://flashinfer.ai/whl/cu121/torch2.4/")
            print("  Continuing with eager attention (slower)")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure device_map based on number of GPUs
        if num_gpus > 1:
            # For multi-GPU, use 'auto' to automatically distribute layers
            device_map_config = 'auto'
            print(f"\nUsing automatic model parallelism across {num_gpus} GPUs")
        else:
            # For single GPU, use simple device placement
            device_map_config = 'auto'
        
        # Load model with automatic device mapping
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map_config,
            trust_remote_code=True,
            attn_implementation="eager",  # Use eager for FlashInfer compatibility
            max_memory={i: f"{torch.cuda.get_device_properties(i).total_memory * 0.9 / 1e9:.0f}GB" 
                       for i in range(num_gpus)} if num_gpus > 1 else None,
        )
        
        self.model.eval()
        
        # Get the device of the first parameter (where inputs should go)
        self.input_device = next(self.model.parameters()).device
        print(f"Model loaded. First layer on: {self.input_device}")
        
        # Print device map if multi-GPU
        if num_gpus > 1:
            print("\nModel distribution:")
            device_map = self.model.hf_device_map
            device_counts = {}
            for layer, device in device_map.items():
                device_str = str(device)
                device_counts[device_str] = device_counts.get(device_str, 0) + 1
            for device, count in sorted(device_counts.items()):
                print(f"  {device}: {count} layers")
        
        # Apply torch.compile for optimization
        if self.compile_model:
            print("\nCompiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode='max-autotune')
        
        # Enable cuDNN benchmark for optimal performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    def benchmark(self, prompt_length=1, decode_length=512, warmup_runs=3, benchmark_runs=10):
        """Run benchmark with specified parameters"""
        
        # Create dummy input with specified prompt length
        dummy_text = "Hello " * prompt_length
        inputs = self.tokenizer(
            [dummy_text] * self.batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=prompt_length
        )
        
        # Move inputs to the device where the first model layer is
        inputs = {k: v.to(self.input_device) for k, v in inputs.items()}
        
        print(f"\nInput shape: {inputs['input_ids'].shape}")
        print(f"Input device: {inputs['input_ids'].device}")
        print(f"Target decode length: {decode_length}")
        
        # Warmup runs
        print(f"\nWarming up ({warmup_runs} runs)...")
        with torch.no_grad():
            for i in range(warmup_runs):
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=decode_length,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                print(f"  Warmup {i+1}/{warmup_runs} completed")
        
        # Benchmark runs
        print(f"\nRunning benchmark ({benchmark_runs} runs)...")
        latencies = []
        
        with torch.no_grad():
            for i in range(benchmark_runs):
                # Synchronize all GPUs before starting
                if self.num_gpus > 1:
                    for gpu_id in range(self.num_gpus):
                        torch.cuda.synchronize(gpu_id)
                else:
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=decode_length,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                # Synchronize all GPUs after completion
                if self.num_gpus > 1:
                    for gpu_id in range(self.num_gpus):
                        torch.cuda.synchronize(gpu_id)
                else:
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                latency = end_time - start_time
                latencies.append(latency)
                
                actual_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
                per_token_latency = latency / (actual_tokens * self.batch_size)
                
                print(f"  Run {i+1}/{benchmark_runs}: {latency:.4f}s "
                      f"({per_token_latency*1000:.4f}ms/token)")
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        actual_decode_length = outputs.shape[1] - inputs['input_ids'].shape[1]
        total_tokens = actual_decode_length * self.batch_size
        per_token_latency = avg_latency / total_tokens
        per_iteration_latency = per_token_latency * self.batch_size
        throughput = total_tokens / avg_latency
        
        results = {
            'model': self.model_name,
            'batch_size': self.batch_size,
            'num_gpus': self.num_gpus,
            'prompt_length': prompt_length,
            'target_decode_length': decode_length,
            'actual_decode_length': actual_decode_length,
            'avg_total_latency_s': avg_latency,
            'per_token_latency_ms': per_token_latency * 1000,
            'per_iteration_latency_ms': per_iteration_latency * 1000,
            'per_token_latency_s': per_token_latency,
            'throughput_tokens_per_s': throughput,
            'gpus': [torch.cuda.get_device_name(i) for i in range(self.num_gpus)],
            'torch_compile': self.compile_model,
            'attention_backend': 'flashinfer',
            'all_latencies': latencies,
        }
        
        return results

def main():
    parser = argparse.ArgumentParser(description='LLM Inference Benchmark with Multi-GPU Support')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., Qwen/Qwen2.5-8B)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='Number of GPUs to use (model will be automatically distributed)')
    parser.add_argument('--prompt-length', type=int, default=1,
                       help='Input prompt length in tokens')
    parser.add_argument('--decode-length', type=int, default=512,
                       help='Number of tokens to generate')
    parser.add_argument('--warmup-runs', type=int, default=10,
                       help='Number of warmup runs')
    parser.add_argument('--benchmark-runs', type=int, default=10,
                       help='Number of benchmark runs')
    parser.add_argument('--no-compile', action='store_true',
                       help='Disable torch.compile()')
    parser.add_argument('--output', type=str, default='results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")
    
    # Verify enough GPUs are available
    num_gpus = torch.cuda.device_count()
    if args.num_gpus > num_gpus:
        raise RuntimeError(f"Requested {args.num_gpus} GPUs but only {num_gpus} available")
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    print()
    
    # Initialize benchmark
    benchmark = LLMBenchmark(
        model_name=args.model,
        batch_size=args.batch_size,
        compile_model=not args.no_compile,
        num_gpus=args.num_gpus
    )
    
    # Run benchmark
    results = benchmark.benchmark(
        prompt_length=args.prompt_length,
        decode_length=args.decode_length,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Model: {results['model']}")
    print(f"Batch Size: {results['batch_size']}")
    print(f"Number of GPUs: {results['num_gpus']}")
    print(f"GPUs: {', '.join(results['gpus'])}")
    print(f"Attention Backend: {results['attention_backend']}")
    print(f"Prompt Length: {results['prompt_length']}")
    print(f"Decode Length: {results['actual_decode_length']}")
    print(f"Average Total Latency: {results['avg_total_latency_s']:.4f}s")
    print(f"Per-Token Latency: {results['per_token_latency_ms']:.4f}ms")
    print(f"Throughput: {results['throughput_tokens_per_s']:.2f} tokens/s")
    print(f"{'='*60}\n")
    
    # Save results
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    all_results.append(results)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()