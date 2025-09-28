import torch
import runtime_kernel_cute_hopper

# torch.set_printoptions(sci_mode=False, profile="full")
torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)

reduction_size = 4096
output_size = 16
batch_size = 64

x = torch.randn((batch_size, reduction_size), device="cuda", dtype=torch.bfloat16)
w = torch.randn(
    (reduction_size, output_size), device="cuda", dtype=torch.bfloat16
)
residual = torch.randn(batch_size, output_size, device="cuda", dtype=torch.bfloat16)
# residual = torch.full((batch_size, output_size), 1, device="cuda", dtype=torch.bfloat16)

output = torch.empty(batch_size, output_size, device="cuda", dtype=torch.bfloat16)


runtime_kernel_cute_hopper.linear(x, w, residual, output)

print("output from kernel:")

torch_out = torch.matmul(x, w)
torch_out = torch_out + residual
print(output)
print(torch_out)