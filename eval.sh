vllm bench latency --model Qwen/Qwen3-8B --input-len 1 --output-len 512 --batch-size 1 2 4 8 16

python -m sglang.bench_one_batch --model-path Qwen/Qwen3-1.7B --batch 16 --input-len 1 --output-len 512

ython /home/ubuntu/a100/mirage/demo/qwen3/demo.py --use-mirage --max-num-batched-requests 16 --max-num-batched-tokens 16 --model Qwen/Qwen3-8B

python -m sglang.bench_one_batch --model-path meta-llama/Llama-3.2-1B-Instruct --batch 1 2 4 8  16 --input-len 1 --output-len 512

python /home/ubuntu/a100/mirage/demo/llama3/demo.py --use-mirage --max-num-batched-requests 4 --max-num-batched-tokens 4 --model meta-llama/Llama-3.2-1B-Instruct

vllm bench latency --model meta-llama/Llama-3.2-1B-Instruct --input-len 1 --output-len 512 --batch-size 16