import torch
import runtime_kernel_hopper

# torch.set_printoptions(sci_mode=False, profile="full")
torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)

reduction_sizes = [4096]
output_sizes = [4096]
batch_size = 64

x = torch.randn((batch_size, reduction_size), device="cuda", dtype=torch.bfloat16)
w = torch.randn(
    (output_size, reduction_size), device="cuda", dtype=torch.bfloat16
)
residual = torch.randn(batch_size, output_size, device="cuda", dtype=torch.bfloat16)
output = torch.empty(batch_size, output_size, device="cuda", dtype=torch.bfloat16)


runtime_kernel_cute_hopper.linear(x, w, residual, output)

print("output from kernel:")

torch_out = torch.matmul(x, torch.transpose(w, 0, 1))
print(output)
print(torch_out)