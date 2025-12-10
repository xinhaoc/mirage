vllm bench latency --model Qwen/Qwen3-8B --input-len 1 --output-len 512 --batch-size 1 2 4 8 16

python -m sglang.bench_one_batch --model-path Qwen/Qwen3-1.7B --batch 16 --input-len 1 --output-len 512

python /home/ubuntu/a100/mirage/demo/qwen3/demo.py --use-mirage --max-num-batched-requests 16 --max-num-batched-tokens 16 --model Qwen/Qwen3-8B

python -m sglang.bench_one_batch --model-path meta-llama/Llama-3.2-1B-Instruct --batch 1 2 4 8  16 --input-len 1 --output-len 512

python /home/ubuntu/a100/mirage/demo/llama3/demo.py --use-mirage --max-num-batched-requests 4 --max-num-batched-tokens 4 --model meta-llama/Llama-3.2-1B-Instruct

vllm bench latency --model meta-llama/Llama-3.2-1B-Instruct --input-len 1 --output-len 512 --batch-size 16


python -m sglang.bench_one_batch --model-path Qwen/Qwen3-8B --batch 1 2 4 8 16 --input-len 1 --output-len 512


vllm bench latency --model Qwen/Qwen3-1.7B --input-len 1 --output-len 512 --batch-size 1
vllm bench latency --model Qwen/Qwen3-1.7B --input-len 1 --output-len 512 --batch-size 2
vllm bench latency --model Qwen/Qwen3-1.7B --input-len 1 --output-len 512 --batch-size 4
vllm bench latency --model Qwen/Qwen3-1.7B --input-len 1 --output-len 512 --batch-size 8
vllm bench latency --model Qwen/Qwen3-1.7B --input-len 1 --output-len 512 --batch-size 16

vllm bench latency --model meta-llama/Llama-3.2-1B-Instruct --input-len 1 --output-len 512 --batch-size 1
vllm bench latency --model meta-llama/Llama-3.2-1B-Instruct --input-len 1 --output-len 512 --batch-size 2
vllm bench latency --model meta-llama/Llama-3.2-1B-Instruct --input-len 1 --output-len 512 --batch-size 4
vllm bench latency --model meta-llama/Llama-3.2-1B-Instruct --input-len 1 --output-len 512 --batch-size 8
vllm bench latency --model meta-llama/Llama-3.2-1B-Instruct --input-len 1 --output-len 512 --batch-size 16


nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --capture-range=nvtx \
  --capture-range-end=stop \
  -o sglang_prefill \
  python -m sglang.bench_one_batch \
    --model-path Qwen/Qwen3-8B
    --batch-size 1 \
    --input-len 1024 \
    --output-len 32 \
    --profile \
    --profile-activities CUDA_PROFILER


nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  -o sglang_prefill \
  python -m sglang.bench_one_batch \
    --model-path Qwen/Qwen3-8B \
    --batch-size 1 \
    --input-len 1 \
    --output-len 512 \
    --tp-size 2conda activate torch


    ~/hh1001/mirage_xc/vllm_profile/Qwen_Qwen3-8B/Dec7

    ~/hh1001/mirage_xc
vllm bench latency --model Qwen/Qwen3-30B-A3B --input-len 1 --output-len 512 --batch-size 1

    python demo/qwen3/demo_30B_A3B_hopper.py --use-mirage --max-num-batched-requests 1  --model Qwen/Qwen3-30B-A3B
python -m sglang.bench_one_batch --model-path Qwen/Qwen3-30B-A3B --batch 16 --input-len 1 --output-len 512
