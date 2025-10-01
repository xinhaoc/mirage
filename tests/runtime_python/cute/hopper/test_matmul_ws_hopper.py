import torch
import runtime_kernel_cute_hopper

# torch.set_printoptions(sci_mode=False, profile="full")
torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)

reduction_size = 4096
batch_size = 16
output_size = 64

weight = torch.randn((output_size, reduction_size), device="cuda", dtype=torch.bfloat16, generator=g)
x = torch.randn(
    (reduction_size, batch_size), device="cuda", dtype=torch.bfloat16, generator=g
)
residual = torch.zeros(batch_size, output_size, device="cuda", dtype=torch.bfloat16)

output = torch.empty(output_size, batch_size, device="cuda", dtype=torch.bfloat16)

runtime_kernel_cute_hopper.linear(weight, x, residual, output)

output_reshape = output.reshape(batch_size, output_size)

# torch_out = torch.matmul(weight, x)
torch_out = torch.matmul(x.T, weight.T)
torch_out = torch_out + residual
print(output.shape)
print(torch_out.shape)
print("output_reshape from kernel:")
print(output_reshape)
print(torch_out)
print(torch.allclose(output_reshape, torch_out, atol=1e-2, rtol=1e-2))