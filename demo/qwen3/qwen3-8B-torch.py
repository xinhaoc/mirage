import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure per-token decode latency for Qwen-style models on H100"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Model name or path, e.g. "
            "'Qwen/Qwen2-7B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', "
            "or local path to Qwen3-8B."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the input prompts.",
    )

    parser.add_argument(
        "--num-new-tokens",
        type=int,
        default=511,
        help="Number of tokens to generate in the decode benchmark.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=".",
        help="Prompt text to use (will be repeated for the whole batch).",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Inference precision.",
    )

    parser.add_argument(
        "--use-compile",
        action="store_true",
        help="Use torch.compile for the model (PyTorch 2.x, adds compile overhead but speeds up steady-state).",
    )

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the model (often needed for Qwen).",
    )

    return parser.parse_args()


def get_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def main():
    args = parse_args()

    # Basic CUDA/H100-friendly settings
    assert torch.cuda.is_available(), "CUDA is required to run this script."
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    dtype = get_dtype(args.dtype)

    print(f"Loading tokenizer and model from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    # Some Qwen models don't have pad_token set -> make it equal to eos.
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # Fallback: add a pad token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            print("Added [PAD] token as pad_token_id.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    # If we added tokens, resize embeddings
    if len(tokenizer) != model.get_input_embeddings().weight.size(0):
        model.resize_token_embeddings(len(tokenizer))

    if args.use_compile:
        print("Compiling model with torch.compile(...) (this may take a while)...")
        model = torch.compile(model, mode="max-autotune")

    # Build batch of prompts
    prompts = [args.prompt] * args.batch_size
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    print(f"\nPrompt batch shape: input_ids={inputs['input_ids'].shape}")
    print(f"Batch size: {args.batch_size}")
    print(f"New tokens (decode steps): {args.num_new_tokens}")
    print(f"Precision: {args.dtype}")
    print(f"Using torch.compile: {args.use_compile}")

    # --------------------------------------------------------
    # 1) Prefill phase: run full prompt once to build KV cache
    # --------------------------------------------------------
    print("\nRunning prefill to build KV cache (not timed)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    prefill_out = model(**inputs, use_cache=True)
    past_key_values = prefill_out.past_key_values

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    prefill_time = t1 - t0
    total_prompt_tokens = int(inputs["input_ids"].ne(tokenizer.pad_token_id).sum())
    print(
        f"Prefill done: {prefill_time:.4f} s "
        f"({total_prompt_tokens / prefill_time:.1f} tokens/s over batch)."
    )

    # Initial next token for decode loop: take last token from each sequence
    next_token = inputs["input_ids"][:, -1:].contiguous()

    # Warm-up a few decode steps to stabilize kernels (not timed)
    print("Running a few warm-up decode steps (not timed)...")
    with torch.no_grad():
        for _ in range(4):
            out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            # Use argmax as a simple decoding strategy just to keep sequence going
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = out.past_key_values

    # Rebuild cache for actual measurement to have consistent starting point
    with torch.no_grad():
        prefill_out = model(**inputs, use_cache=True)
    past_key_values = prefill_out.past_key_values
    next_token = inputs["input_ids"][:, -1:].contiguous()

    # --------------------------------------------------------
    # 2) Decode loop: measure per-step and per-token latency
    # --------------------------------------------------------
    print("\nMeasuring decode (per-token) latency...")
    decode_latencies = []

    with torch.no_grad():
        for step in range(args.num_new_tokens):
            torch.cuda.synchronize()
            t_start = time.perf_counter()

            out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )

            torch.cuda.synchronize()
            t_end = time.perf_counter()

            step_time = t_end - t_start
            decode_latencies.append(step_time)

            # Greedy next token
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = out.past_key_values

    total_decode_time = sum(decode_latencies)
    avg_step_time = total_decode_time / len(decode_latencies)
    # Each step generates batch_size tokens
    per_token_time = avg_step_time / args.batch_size

    tokens_generated = args.batch_size * args.num_new_tokens
    decode_throughput = tokens_generated / total_decode_time

    print("\n================ Decode Latency Report ================")
    print(f"Model:        {args.model}")
    print(f"Device:       {device} (H100 assumed)")
    print(f"Batch size:   {args.batch_size}")
    print(f"Prompt tokens (total over batch): {total_prompt_tokens}")
    print(f"New tokens:   {args.num_new_tokens}")
    print(f"Precision:    {args.dtype}")
    print(f"torch.compile: {args.use_compile}")
    print("------------------------------------------------------")
    print(f"Avg step time:        {avg_step_time*1000:.3f} ms/step "
          f"(for {args.batch_size} tokens)")
    print(f"Per-token latency:    {per_token_time*1000:.3f} ms/token")
    print(f"Decode throughput:    {decode_throughput:.1f} tokens/s")
    print("======================================================\n")


if __name__ == "__main__":
    main()
