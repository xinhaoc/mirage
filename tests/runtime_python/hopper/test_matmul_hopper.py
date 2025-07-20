import torch
import runtime_kernel_hopper


input = torch.randn(64, 4096, dtype=torch.bfloat16, device='cuda')
weight = torch.randn(4096, 64, dtype=torch.bfloat16, device='cuda')
output = torch.empty(64, 64, dtype=torch.bfloat16, device='cuda')

runtime_kernel_hopper.linear_kernel(input, weight, output)
print(output)