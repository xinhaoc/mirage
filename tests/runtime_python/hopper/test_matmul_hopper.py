import torch
import runtime_kernel_hopper

torch.set_printoptions(sci_mode=False, profile="full")

g = torch.Generator(device='cuda').manual_seed(1234)

# reduction_size = 4096
# output_sizes = [16, 32, 64]
reduction_size = 128
output_sizes = [64]

for output_size in output_sizes:
    print(f"\n=== Testing output_size = {output_size} ===")

    # x = torch.ones((64, reduction_size), device="cuda", dtype=torch.bfloat16)
    # w = torch.ones((output_size, reduction_size), device="cuda", dtype=torch.bfloat16)

    x32 = torch.empty((64, reduction_size), device="cuda", dtype=torch.float32)
    torch.nn.init.trunc_normal_(x32, mean=0.0, std=0.05, a=-0.1, b=0.1)  # 先采样再截断
    x = x32.to(torch.bfloat16)

    w32 = torch.empty((output_size, reduction_size), device="cuda", dtype=torch.float32)
    torch.nn.init.trunc_normal_(w32, mean=0.0, std=0.05, a=-0.1, b=0.1)  # 先采样再截断
    w = w32.to(torch.bfloat16)

    # x = torch.randn((64, reduction_size), device='cuda', dtype=torch.bfloat16, generator=g)
    # w = torch.randn((output_size, reduction_size), device='cuda', dtype=torch.bfloat16, generator=g)

    # x = torch.randn((64, reduction_size), device="cuda", dtype=torch.bfloat16)
    # w = torch.randn((output_size, reduction_size), device="cuda", dtype=torch.bfloat16)
    
    output = torch.empty(64, output_size, device="cuda", dtype=torch.bfloat16)


    # for i in range(64):
    #     for j in range(reduction_size):
    #         # if i == 4 and j == 4:
    #         #     x[i, j] = 1001
    #         # else:
    #         x[i, j] = 0.0001 + (i * reduction_size + j) * 0.0001

    
    # for i in range(output_size):
    #     for j in range(reduction_size):
    #         if i == 2 and j == 2:
    #             w[i, j] = 5005
    #         else:
    #             w[i, j] = 0.0001 + (i * reduction_size + j) * 0.0001

    # x[1, 0] = 0.2
    # x[3, 0] = 0.3
    # x = torch.round(x, decimals=1)
    # w = torch.round(w, decimals=1)

    runtime_kernel_hopper.linear(x, w, output)
    torch_out = torch.matmul(x, torch.transpose(w, 0, 1))

    print("torch_out.shape", torch_out.shape)
    print(torch_out)
    print("output.shape", output.shape)
    print(output)

    # print all of x
    # print("x.shape", x.shape)
    # print(x)

    # result of [4, 0] of x matmul w
    print("result of [4, 0] of x matmul w")
    print(sum(x[4, i] * w[0, i] for i in range(reduction_size)))
    print('result of [4, 1] of x matmul w')
    print(sum(x[4, i] * w[1, i] for i in range(reduction_size)))
    print('result of [4, 2] of x matmul w')
    print(sum(x[4, i] * w[2, i] for i in range(reduction_size)))
    print('result of [4, 3] of x matmul w')
    print(sum(x[4, i] * w[3, i] for i in range(reduction_size)))
    print('result of [4, 4] of x matmul w')
    print(sum(x[4, i] * w[4, i] for i in range(reduction_size)))


    print("Ratio (kernel / torch):")
    print(output / torch_out)

    ok = torch.allclose(output.float(), torch_out.float(), rtol=1e-2, atol=1e-2)
    max_abs = (output.float() - torch_out.float()).abs().max()
    max_rel = ((output.float() - torch_out.float()).abs() / (torch_out.float().abs() + 1e-6)).max()
    print(ok, max_abs.item(), max_rel.item())
