import torch
import runtime_kernel_fp8_blackwell

torch.set_printoptions(sci_mode=False, profile="full")


def quantize_to_fp8_e4m3(tensor_bf16):
    """Quantize BF16 tensor to FP8 E4M3 with per-block (128) scale factors."""
    # FP8 E4M3 max value
    fp8_max = 448.0
    shape = tensor_bf16.shape
    assert len(shape) == 2
    rows, cols = shape

    # Pad cols to multiple of 128
    sf_cols = (cols + 127) // 128
    padded_cols = sf_cols * 128

    t = tensor_bf16.float()
    if padded_cols > cols:
        t = torch.nn.functional.pad(t, (0, padded_cols - cols))

    # Reshape to blocks of 128
    t_blocks = t.reshape(rows, sf_cols, 128)

    # Compute per-block amax
    amax = t_blocks.abs().amax(dim=-1)  # [rows, sf_cols]
    amax = amax.clamp(min=1e-12)

    # Compute scale = amax / fp8_max, stored as E8M0 (power-of-2)
    # E8M0: log2(scale) rounded to nearest int, stored as biased exponent
    log2_scale = torch.ceil(torch.log2(amax / fp8_max))
    log2_scale = log2_scale.clamp(min=-127, max=127)
    # E8M0 biased exponent: bias = 127
    sf_uint8 = (log2_scale.int() + 127).clamp(0, 254).byte()  # [rows, sf_cols]

    # Actual scale values (power of 2)
    scale = torch.pow(2.0, log2_scale)  # [rows, sf_cols]

    # Quantize: fp8_val = clamp(round(bf16_val / scale), -fp8_max, fp8_max)
    scale_expanded = scale.unsqueeze(-1).expand_as(t_blocks)
    t_scaled = t_blocks / scale_expanded
    t_clamped = t_scaled.clamp(-fp8_max, fp8_max)
    t_fp8 = t_clamped.to(torch.float8_e4m3fn)

    # Reshape back (drop padding)
    t_fp8 = t_fp8.reshape(rows, padded_cols)[:, :cols].contiguous()

    return t_fp8, sf_uint8, scale


def dequantize_fp8(t_fp8, scale, original_cols):
    """Dequantize FP8 tensor back to FP32 using block scales."""
    rows = t_fp8.shape[0]
    cols = t_fp8.shape[1]
    sf_cols = (cols + 127) // 128
    padded_cols = sf_cols * 128

    t = t_fp8.float()
    if padded_cols > cols:
        t = torch.nn.functional.pad(t, (0, padded_cols - cols))

    t_blocks = t.reshape(rows, sf_cols, 128)
    scale_expanded = scale.unsqueeze(-1).expand_as(t_blocks)
    t_deq = t_blocks * scale_expanded

    return t_deq.reshape(rows, padded_cols)[:, :original_cols]


g = torch.Generator(device="cuda").manual_seed(1234)

batch_size = 1
output_size = 128
reduction_size = 768

print(
    f"\n=== Testing FP8 GEMM: batch={batch_size} output={output_size} "
    f"reduction={reduction_size} ==="
)

# Create random BF16 tensors
x_bf16 = torch.randn(
    (batch_size, reduction_size), device="cuda", dtype=torch.bfloat16, generator=g
)
w_bf16 = torch.randn(
    (output_size, reduction_size), device="cuda", dtype=torch.bfloat16, generator=g
)

# Quantize to FP8 with block scale factors
x_fp8, sfb_uint8, sfb_scale = quantize_to_fp8_e4m3(x_bf16)
w_fp8, sfa_uint8, sfa_scale = quantize_to_fp8_e4m3(w_bf16)

# SFA: scale for A (weight) [ceil(output/128), ceil(reduction/128)]
# SFB: scale for B (input)  [ceil(batch/128), ceil(reduction/128)]
print(f"x_fp8 shape: {x_fp8.shape}, sfb shape: {sfb_uint8.shape}")
print(f"w_fp8 shape: {w_fp8.shape}, sfa shape: {sfa_uint8.shape}")

# Reinterpret FP8 as uint8 for the kernel
x_uint8 = x_fp8.view(torch.uint8)
w_uint8 = w_fp8.view(torch.uint8)

# Output tensor (BF16)
output = torch.empty(batch_size, output_size, device="cuda", dtype=torch.bfloat16)

# Run kernel
runtime_kernel_fp8_blackwell.fp8_linear_sm100_mpk(
    x_uint8, w_uint8, sfa_uint8.contiguous(), sfb_uint8.contiguous(), output
)

# Reference: dequantize(input) @ dequantize(weight).T
x_deq = dequantize_fp8(x_fp8, sfb_scale.cuda(), reduction_size)
w_deq = dequantize_fp8(w_fp8, sfa_scale.cuda(), reduction_size)
ref_output = torch.matmul(x_deq, w_deq.t()).to(torch.bfloat16)

print(f"Kernel output: {output}")
print(f"Reference:     {ref_output}")
print(f"Max diff: {(output.float() - ref_output.float()).abs().max().item():.6f}")

torch.testing.assert_close(
    output,
    ref_output,
    rtol=5e-2,
    atol=5e-2,
)
print("Test passed!")

# Warm-up
for _ in range(16):
    runtime_kernel_fp8_blackwell.fp8_linear_sm100_mpk(
        x_uint8, w_uint8, sfa_uint8, sfb_uint8, output
    )

torch.cuda.synchronize()
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
    enable_timing=True
)
repetitions = 1000
starter.record()
for rep in range(repetitions):
    runtime_kernel_fp8_blackwell.fp8_linear_sm100_mpk(
        x_uint8, w_uint8, sfa_uint8, sfb_uint8, output
    )
ender.record()
torch.cuda.synchronize()
total_time = starter.elapsed_time(ender)
avg_time = total_time / repetitions
print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")
