// Ampere task impls
#include "tasks/ampere/embedding.cuh"
#include "tasks/ampere/silu_mul.cuh"
// Hopper task impls
#include "tasks/cute/hopper/gemm_ws.cuh"
#include "tasks/cute/hopper/gemm_ws_cooperative.cuh"
#include "tasks/cute/hopper/gemm_ws_mpk.cuh"
#include "tasks/hopper/linear_hopper.cuh"
#include "tasks/hopper/linear_swapAB_hopper.cuh"
#include "tasks/hopper/multitoken_paged_attention_hopper.cuh"
#include "tasks/hopper/rmsnorm_hopper.cuh"
#include "tasks/hopper/silu_mul_hopper.cuh"
// Blackwell task impls
#include "argmax_sm100.cuh"
#include "attention_sm100.cuh"
#include "linear_sm100_mpk.cuh"
#include "moe_linear_sm100.cuh"
#include "mul_sum_add_sm100.cuh"
#include "topk_softmax_sm100.cuh"
