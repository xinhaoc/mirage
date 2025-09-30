#pragma once
// CUTE
#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>

// CUTLASS
#include <cutlass/cutlass.h>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/arch/grid_dependency_control.h>
// #include <cutlass/util/packed_stride.hpp>   // ← 你缺这个，导致 make_cute_packed_stride 未找到

// 你的 kernel 类型与设备内核（提供 MMAKernelTraits / Collective* / gemm_kernel_*）
#include "tasks/cute/hopper/kernel_traits.cuh"
#include "tasks/cute/hopper/mma_tma_ws_mainloop.cuh"
#include "tasks/cute/hopper/epilogue.cuh"
#include "tasks/cute/hopper/gemm_ws.cuh"
