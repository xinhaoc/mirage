#!/bin/bash

# Simple script to run all benchmarks
# Models: Qwen3-8B, Qwen3-1.7B, Qwen3-30B-A3B, Qwen3-0.6B, Llama-3.2-1B-Instruct
# Batch sizes: 1, 2, 4, 8

set -e  # Exit on error

# Configuration
SCRIPT="demo/qwen3/torch_baseline.py"
OUTPUT_FILE="benchmark_results.json"
PROMPT_LENGTH=1
DECODE_LENGTH=512

# Models to test (UPDATE THESE PATHS TO MATCH YOUR HUGGINGFACE MODEL NAMES)

MODELS=(
    # "Qwen/Qwen3-8B"           # Qwen3-8B
    # "Qwen/Qwen3-1.7B"         # Qwen3-1.7B
    # "Qwen/Qwen3-0.6B"          # Qwen3-30B
    # "Qwen/Qwen3-30B-A3B"         # Qwen3-0.6B
    "meta-llama/Llama-3.2-1B-Instruct"  # Llama-3.2-1B
)

# Batch sizes to test
BATCH_SIZES=(1 2 4 8)

# Get GPU info
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "======================================"
echo "GPU: $GPU_NAME"
echo "======================================"
echo ""

# Create timestamped output file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="results_${TIMESTAMP}.json"

echo "Results will be saved to: $OUTPUT_FILE"
echo ""

# Calculate total experiments
TOTAL=$((${#MODELS[@]} * ${#BATCH_SIZES[@]}))
CURRENT=0

# Run experiments
for model in "${MODELS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        echo ""
        echo "======================================"
        echo "Progress: $CURRENT/$TOTAL"
        echo "Model: $model"
        echo "Batch Size: $bs"
        echo "======================================"
        
        # Run benchmark
        python $SCRIPT \
            --model "$model" \
            --batch-size $bs \
            --prompt-length $PROMPT_LENGTH \
            --decode-length $DECODE_LENGTH \
            --output "$OUTPUT_FILE" \
            2>&1 | tee -a benchmark_log.txt
        
        if [ $? -eq 0 ]; then
            echo "✓ Success"
        else
            echo "✗ Failed - continuing to next experiment"
        fi
        
        # Small delay between experiments
        sleep 2
    done
done

echo ""
echo "======================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "======================================"
echo "Results saved to: $OUTPUT_FILE"
echo "Log saved to: benchmark_log.txt"
echo ""

# Generate summary if Python is available
if command -v python &> /dev/null; then
    echo "Generating summary..."
    python << 'PYEOF'
import json
import sys

try:
    with open(sys.argv[1], 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Model':<35} {'BS':<5} {'Per-Token (ms)':<16} {'Throughput (tok/s)':<20}")
    print("-"*80)
    
    for r in results:
        model_short = r['model'].split('/')[-1]
        print(f"{model_short:<35} {r['batch_size']:<5} {r['per_token_latency_ms']:<16.4f} {r['throughput_tokens_per_s']:<20.2f}")
    
    print("="*80)
    
    # Save CSV
    csv_file = sys.argv[1].replace('.json', '.csv')
    with open(csv_file, 'w') as f:
        f.write("Model,Batch_Size,Per_Token_Latency_ms,Throughput_tokens_per_s,GPU\n")
        for r in results:
            f.write(f"{r['model']},{r['batch_size']},{r['per_token_latency_ms']:.4f},{r['throughput_tokens_per_s']:.2f},{r['gpu']}\n")
    
    print(f"\nCSV saved to: {csv_file}")
    
except Exception as e:
    print(f"Could not generate summary: {e}")
PYEOF
    python -c "import sys; sys.argv.append('$OUTPUT_FILE')" "$OUTPUT_FILE"
fi

echo ""
echo "Done!"