import torch
import runtime_kernel_cute_hopper

# torch.set_printoptions(sci_mode=False, profile="full")
torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)

reduction_size = 4096
batch_size = 16
output_size = 64

x = torch.randn(
    (reduction_size, batch_size), device="cuda", dtype=torch.bfloat16, generator=g
)
weight = torch.randn((output_size, reduction_size), device="cuda", dtype=torch.bfloat16, generator=g)
for i in range(reduction_size):
    for j in range(batch_size):
        x[i, j] = 0.2
for i in range(output_size):
    for j in range(reduction_size):
        weight[i, j] = 0.1

# x = torch.randn((batch_size, reduction_size), device="cuda", dtype=torch.bfloat16, generator=g)
residual = torch.zeros(batch_size, output_size, device="cuda", dtype=torch.bfloat16)

output = torch.empty(batch_size, output_size, device="cuda", dtype=torch.bfloat16)

runtime_kernel_cute_hopper.linear(weight, x, residual, output)

# torch_out = torch.matmul(weight, x)
torch_out = torch.matmul(x.T, weight.T)
torch_out = torch_out + residual
print(output.shape)
print(torch_out.shape)
print("output from kernel:")
print(output)
print(torch_out)
print(torch.allclose(output, torch_out, atol=1e-2, rtol=1e-2))