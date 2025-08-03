import torch
import runtime_kernel_hopper

torch.set_printoptions(sci_mode=False)

# reduction_size = 4096
# output_sizes = [16, 32, 64]
reduction_size = 4096
output_sizes = [32]

for output_size in output_sizes:
    print(f"\n=== Testing output_size = {output_size} ===")
    x = torch.randn((64, reduction_size), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((output_size, reduction_size), device="cuda", dtype=torch.bfloat16)
    output = torch.empty(64, output_size, device="cuda", dtype=torch.bfloat16)

    runtime_kernel_hopper.linear(x, w, output)
    torch_out = torch.matmul(x, torch.transpose(w, 0, 1))

    print("Ratio (kernel / torch):")
    print(output / torch_out)
