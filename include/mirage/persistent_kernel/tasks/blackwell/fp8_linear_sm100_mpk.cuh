#pragma once
#include <cstdio>
#include <iostream>

// Cutlass includes
#include <cutlass/half.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

// CuTe includes
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>

#include "../common/dmem_layout.cuh"
#include "../common/worker_config.h"
#include "../hopper/barrier.cuh"
#include "../hopper/smem_layout_tma.cuh"
#include "../hopper/tma.cuh"
#include "storage.cuh"

namespace kernel {

// FP8 block-scaled GEMM for SM100 (Blackwell).
// Adapted from linear_sm100_mpk_task_impl for FP8 E4M3 with E8M0 block scaling.
//
// A: weight [OUTPUT_SIZE, REDUCTION_SIZE] in FP8
// B: input  [BATCH_SIZE, REDUCTION_SIZE] in FP8
// SFA: scale factors for A [ceil(OUTPUT_SIZE/128), ceil(REDUCTION_SIZE/128)] uint8
// SFB: scale factors for B [ceil(BATCH_SIZE/128), ceil(REDUCTION_SIZE/128)] uint8
// Output: [BATCH_SIZE, OUTPUT_SIZE] in BF16
//
// Warp roles: 5=TMA, 4=MMA, 0-3=Epilogue, 6-7=idle
template <typename FP8Type_,
          typename TMA_A,
          typename TMA_B,
          typename TMA_OUT,
          int MMA_M,
          int MMA_N,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int NUM_AB_STAGE = 8,
          int NUM_ACC_STAGE = 2,
          int NUM_C_STAGE = 4>
__device__ __noinline__ void
    fp8_linear_sm100_mpk_task_impl(const TMA_A &tma_a,
                                    const TMA_B &tma_b,
                                    const TMA_OUT &tma_out,
                                    const uint8_t *sfa_ptr,
                                    const uint8_t *sfb_ptr) {
  int warp_idx = cutlass::canonical_warp_idx_sync();

  // Debug: verify fresh build
  if (threadIdx.x == 0) {
    printf("[FP8 kernel v4] UTCCP-based SF loading\n");
  }

  using OutType = cute::bfloat16_t;

  // ---------------------------------------------------------------
  // MMA Setup: Block-scaled FP8 MMA (SM100_MMA_MXF8F6F4_SS)
  // K=32 per MMA, bK = 4*32 = 128
  // ---------------------------------------------------------------
  auto mma_coord_vmnk =
      cute::make_coord(0, cute::_, cute::_, cute::_);

  cute::TiledMMA tiled_mma = cute::make_tiled_mma(
      cute::SM100_MMA_MXF8F6F4_SS<FP8Type_,
                                   FP8Type_,
                                   float,
                                   cutlass::float_ue8m0_t,
                                   MMA_M,
                                   MMA_N,
                                   cute::UMMA::Major::K,
                                   cute::UMMA::Major::K>{});

  auto bM = cute::tile_size<0>(tiled_mma);
  auto bN = cute::tile_size<1>(tiled_mma);
  auto bK = cute::tile_size<2>(tiled_mma) *
            cute::Int<4>{}; // K=32 * 4 = 128

  auto mma_tiler = cute::make_shape(bM, bN, bK);

  // TMEM: accumulator + SF columns
  // UTCCP 32x128b writes 128 bits per DP = 4 TMEM columns per operation.
  // We need 4 columns for SFA + 4 columns for SFB.
  constexpr int num_acc_columns = MMA_N * NUM_ACC_STAGE;
  constexpr int num_sf_columns = 8; // 4 for SFA + 4 for SFB
  constexpr int num_tmem_columns = num_acc_columns + num_sf_columns;

  // ---------------------------------------------------------------
  // Coordinate Tensors and Tiling
  // ---------------------------------------------------------------
  auto mma_coord = cute::select<1, 2, 3>(mma_coord_vmnk);
  auto cd_tiler = cute::make_shape(bN, bM, bK);

  cute::Tensor mA = cute::make_coord_tensor(cute::make_layout(
      cute::make_shape(OUTPUT_SIZE, REDUCTION_SIZE),
      cute::make_stride(cute::E<1>{}, cute::E<0>{})));
  cute::Tensor mB = cute::make_coord_tensor(cute::make_layout(
      cute::make_shape(BATCH_SIZE, REDUCTION_SIZE),
      cute::make_stride(cute::E<1>{}, cute::E<0>{})));
  cute::Tensor mC = cute::make_coord_tensor(cute::make_layout(
      cute::make_shape(BATCH_SIZE, OUTPUT_SIZE),
      cute::make_stride(cute::E<1>{}, cute::E<0>{})));

  cute::Tensor gA = cute::local_tile(
      mA, mma_tiler, mma_coord,
      cute::Step<cute::_1, cute::X, cute::_1>{});
  cute::Tensor gB = cute::local_tile(
      mB, mma_tiler, mma_coord,
      cute::Step<cute::X, cute::_1, cute::_1>{});
  cute::Tensor gC = cute::local_tile(
      mC, cd_tiler, mma_coord,
      cute::Step<cute::_1, cute::_1, cute::X>{});

  // ---------------------------------------------------------------
  // Shared Memory Layouts
  // ---------------------------------------------------------------
  auto mma_shape_A = cute::partition_shape_A(
      tiled_mma,
      cute::make_shape(cute::Int<MMA_M>{}, cute::size<2>(mma_tiler),
                       cute::Int<NUM_AB_STAGE>{}));
  auto mma_shape_B = cute::partition_shape_B(
      tiled_mma,
      cute::make_shape(cute::Int<MMA_N>{}, cute::size<2>(mma_tiler),
                       cute::Int<NUM_AB_STAGE>{}));
  auto mma_shape_C = cute::make_shape(
      cute::make_shape(cute::Int<MMA_N>{}, cute::Int<MMA_M>{}),
      cute::Int<1>{}, cute::Int<1>{}, cute::Int<NUM_C_STAGE>{});

  // FP8 uses K-major swizzled layout (same atom as BF16 but for FP8 type)
  auto sA_layout = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<FP8Type_>{}, mma_shape_A);
  auto sB_layout = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<FP8Type_>{}, mma_shape_B);

  // Output (BF16) layout — unswizzled
  auto sC_layout_fake = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_INTER_Atom<OutType>{}, mma_shape_C);
  auto sC_shape = cute::make_shape(
      cute::make_shape(cute::Int<MMA_N>{}, cute::Int<MMA_M>{}),
      cute::Int<1>{}, cute::Int<1>{},
      cute::make_shape(cute::Int<1>{}, cute::Int<NUM_C_STAGE>{}));
  auto sC_stride = cute::make_stride(
      cute::make_stride(cute::Int<MMA_M>{}, cute::Int<1>{}),
      cute::Int<0>{}, cute::Int<0>{},
      cute::make_stride(cute::Int<0>{}, cute::Int<MMA_M * MMA_N>{}));
  auto sC_layout = cute::composition(sC_layout_fake.layout_a(),
                                     sC_layout_fake.offset(),
                                     cute::make_layout(sC_shape, sC_stride));

  using SharedStorage =
      PipedSharedStorage<FP8Type_,
                         FP8Type_,
                         OutType,
                         decltype(sA_layout),
                         decltype(sB_layout),
                         decltype(sC_layout),
                         NUM_AB_STAGE,
                         NUM_ACC_STAGE>;

  extern __shared__ char shared_memory[];
  uintptr_t aligned_smem =
      (reinterpret_cast<uintptr_t>(shared_memory) + 127) / 128 * 128;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(aligned_smem);

  // SF staging buffers for UTCCP (512 bytes each, 16-byte aligned).
  // UTCCP 32x128b reads 32 DPs × 16 bytes = 512 bytes from SMEM.
  uintptr_t sf_smem_base = aligned_smem + sizeof(SharedStorage);
  sf_smem_base = (sf_smem_base + 15) & ~(uintptr_t)15;
  uint8_t *sf_smem_sfa = reinterpret_cast<uint8_t *>(sf_smem_base);
  uint8_t *sf_smem_sfb = sf_smem_sfa + 512;

  // ---------------------------------------------------------------
  // Initialize Barriers
  // ---------------------------------------------------------------
  if (warp_idx == 0) {
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        NUM_AB_STAGE>(shared_storage.ab_full_mbar_ptr, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        NUM_AB_STAGE>(shared_storage.ab_empty_mbar_ptr, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        NUM_ACC_STAGE>(shared_storage.acc_full_mbar_ptr, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        NUM_ACC_STAGE>(shared_storage.acc_empty_mbar_ptr, 4);
  }

  cutlass::arch::NamedBarrier tmem_allocation_result_barrier(
      32 + 128, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
  cutlass::arch::NamedBarrier epilogue_wg_barrier(
      128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

  // ---------------------------------------------------------------
  // Smem Tensors and MMA Partitioning
  // ---------------------------------------------------------------
  cute::Tensor tCsA = shared_storage.tensor_sA();
  cute::Tensor tCsB = shared_storage.tensor_sB();
  cute::Tensor sC_epi = shared_storage.tensor_sC();

  auto mma_v = cute::get<0>(mma_coord_vmnk);
  cute::ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
  cute::Tensor tCgA = cta_mma.partition_A(gA);
  cute::Tensor tCgB = cta_mma.partition_B(gB);

  // TMA transaction bytes for FP8 data (1 byte per element)
  int tma_transaction_bytes =
      sizeof(FP8Type_) * cute::size<1>(mma_tiler) * cute::size<2>(mma_tiler) +
      sizeof(FP8Type_) * cute::size<0>(mma_tiler) * cute::size<2>(mma_tiler);

  constexpr int TILE_SIZE = 128; // bK for FP8
  constexpr int INPUT_TMA_TILE_SIZE = 128;
  constexpr int WEIGHT_TMA_TILE_SIZE = 128;
  constexpr int OUTPUT_ATOM_SIZE = 128;
  constexpr int B_param = 3;
  constexpr int M_param = 3;
  constexpr int S_param = 3;

  FP8Type_ *shared_weight = shared_storage.A.begin();
  FP8Type_ *shared_input = shared_storage.B.begin();
  OutType *mm_output = shared_storage.C.begin();

  Barrier *ab_full_mbar_ptr =
      reinterpret_cast<Barrier *>(shared_storage.ab_full_mbar_ptr);

  using InputSmem =
      smem_tma<FP8Type_, B_param, M_param, S_param, MMA_N,
               INPUT_TMA_TILE_SIZE, 1>;
  InputSmem input_smem(shared_input);

  using WeightSmem =
      smem_tma<FP8Type_, B_param, M_param, S_param, OUTPUT_ATOM_SIZE,
               WEIGHT_TMA_TILE_SIZE, 1>;
  WeightSmem input_weight_smem(shared_weight);

  using OutputSmem =
      smem_tma<OutType, 0, M_param, S_param, MMA_N, OUTPUT_ATOM_SIZE, 1>;
  OutputSmem mm_output_smem(mm_output);

  // MMA Fragments
  cute::Tensor tCrA = cta_mma.make_fragment_A(tCsA);
  cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);
  auto acc_shape = cute::partition_shape_C(
      tiled_mma,
      cute::make_shape(cute::size<0>(mma_tiler), cute::size<1>(mma_tiler),
                       cute::Int<NUM_ACC_STAGE>{}));
  auto tCtAcc = tiled_mma.make_fragment_C(acc_shape);

  cutlass::arch::fence_barrier_init();
  __syncthreads();

  int k_tile_count = cute::size<4>(tCgA);

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  __syncthreads();

  // ===============================================================
  // Warp 5: TMA Loader — loads A(FP8), B(FP8)
  // ===============================================================
  if (warp_idx == 5) {
    int total_k_tile_count = 0;
    for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
      for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {

        int num_prev_k_blk = total_k_tile_count;
        total_k_tile_count += k_tile_count;

        int tma_wr_k_tile = 0;
        int smem_wr_buffer =
            (num_prev_k_blk + tma_wr_k_tile) % NUM_AB_STAGE;
        int tma_wr_ab_empty_phase =
            (num_prev_k_blk + tma_wr_k_tile) / NUM_AB_STAGE % 2 ^ 1;

        bool peek_ab_empty_status = kernel::try_wait_barrier(
            shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
            tma_wr_ab_empty_phase);

        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
          int tma_wr_k_tile_next = tma_wr_k_tile + 1;
          int smem_wr_buffer_next =
              (num_prev_k_blk + tma_wr_k_tile_next) % NUM_AB_STAGE;
          int tma_wr_ab_empty_phase_next =
              smem_wr_buffer_next == 0 ? tma_wr_ab_empty_phase ^ 1
                                       : tma_wr_ab_empty_phase;

          if (!peek_ab_empty_status) {
            cute::wait_barrier(
                shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                tma_wr_ab_empty_phase);
          }

          if (cute::elect_one_sync()) {
            int tma_coords_A[2] = {k_tile * TILE_SIZE,
                                   m_tile * OUTPUT_ATOM_SIZE};
            int tma_coords_B[2] = {k_tile * TILE_SIZE, n_tile * MMA_N};

            input_weight_smem.set_ptr(
                shared_weight +
                smem_wr_buffer * OUTPUT_ATOM_SIZE * TILE_SIZE);
            input_smem.set_ptr(shared_input +
                               smem_wr_buffer * MMA_N * TILE_SIZE);

            cute::set_barrier_transaction_bytes(
                shared_storage.ab_full_mbar_ptr[smem_wr_buffer],
                tma_transaction_bytes);

            tma_a.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer],
                               input_weight_smem.base_ptr, tma_coords_A);
            tma_b.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer],
                               input_smem.base_ptr, tma_coords_B);
          }

          if (tma_wr_k_tile_next < k_tile_count) {
            peek_ab_empty_status = kernel::try_wait_barrier(
                shared_storage.ab_empty_mbar_ptr[smem_wr_buffer_next],
                tma_wr_ab_empty_phase_next);
          }

          tma_wr_k_tile = tma_wr_k_tile_next;
          smem_wr_buffer = smem_wr_buffer_next;
          tma_wr_ab_empty_phase = tma_wr_ab_empty_phase_next;
        } // end for k_tile
      }   // end for n_tile
    }

  }
  // ===============================================================
  // Warp 4: MMA — load SF via UTCCP + block-scaled UMMA
  // ===============================================================
  else if (warp_idx == 4) {
    tmem_allocation_result_barrier.arrive_and_wait();
    tCtAcc.data() = shared_storage.tmem_base_ptr;

    // SF TMEM addresses (after accumulator columns).
    // UTCCP writes 4 columns per op, so SFA starts at +0 and SFB at +4.
    uint32_t tmem_sfa_addr = shared_storage.tmem_base_ptr + num_acc_columns;
    uint32_t tmem_sfb_addr =
        shared_storage.tmem_base_ptr + num_acc_columns + 4;

    // Create minimal TMEM tensors for .with() API
    auto sfa_tmem_ptr =
        cute::make_tmem_ptr<cutlass::float_ue8m0_t>(tmem_sfa_addr);
    auto sfb_tmem_ptr =
        cute::make_tmem_ptr<cutlass::float_ue8m0_t>(tmem_sfb_addr);
    auto tCtSFA =
        cute::make_tensor(sfa_tmem_ptr, cute::make_layout(cute::Int<1>{}));
    auto tCtSFB =
        cute::make_tensor(sfb_tmem_ptr, cute::make_layout(cute::Int<1>{}));

    constexpr int sfa_k_dim = (REDUCTION_SIZE + 127) / 128;
    constexpr int sfb_k_dim = (REDUCTION_SIZE + 127) / 128;

    int total_k_tile_count = 0;
    int num_tiles_executed = 0;
    for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
      for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {

        int acc_buf_idx = num_tiles_executed % NUM_ACC_STAGE;
        auto tCtAcc_Slice = tCtAcc(cute::_, cute::_, cute::_, acc_buf_idx);

        int num_prev_k_blk = total_k_tile_count;
        total_k_tile_count += k_tile_count;

        int mma_rd_k_tile = 0;
        int smem_rd_buffer =
            (num_prev_k_blk + mma_rd_k_tile) % NUM_AB_STAGE;
        int mma_rd_ab_full_phase =
            (num_prev_k_blk + mma_rd_k_tile) / NUM_AB_STAGE % 2;

        bool peek_ab_full_status = kernel::try_wait_barrier(
            shared_storage.ab_full_mbar_ptr[smem_rd_buffer],
            mma_rd_ab_full_phase);

        int acc_empty_phase =
            num_tiles_executed / NUM_ACC_STAGE % 2 ^ 1;
        cute::wait_barrier(
            shared_storage.acc_empty_mbar_ptr[acc_buf_idx],
            acc_empty_phase);

        tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;

        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
          int mma_rd_k_tile_next = mma_rd_k_tile + 1;
          int smem_rd_buffer_next =
              (num_prev_k_blk + mma_rd_k_tile_next) % NUM_AB_STAGE;
          int mma_rd_ab_full_phase_next =
              smem_rd_buffer_next == 0 ? mma_rd_ab_full_phase ^ 1
                                       : mma_rd_ab_full_phase;

          // Wait for AB data
          if (!peek_ab_full_status) {
            cute::wait_barrier(
                shared_storage.ab_full_mbar_ptr[smem_rd_buffer],
                mma_rd_ab_full_phase);
          }

          // Write scale factors from global → SMEM → TMEM via UTCCP.
          // tcgen05.st only writes to 1 subpartition (the executing warp's).
          // UTCCP (tcgen05.cp.32x128b.warpx4) broadcasts to all 4 subpartitions.
          {
            uint32_t sfa_val = static_cast<uint32_t>(
                sfa_ptr[m_tile * sfa_k_dim + k_tile]);
            uint32_t sfb_val = static_cast<uint32_t>(
                sfb_ptr[n_tile * sfb_k_dim + k_tile]);

            uint32_t lane_id = threadIdx.x % 32;

            // Fill SFA SMEM buffer: replicate the SF byte across all 512 bytes
            uint32_t sfa_fill = sfa_val | (sfa_val << 8) |
                                (sfa_val << 16) | (sfa_val << 24);
            uint4 sfa_vec = {sfa_fill, sfa_fill, sfa_fill, sfa_fill};
            reinterpret_cast<uint4 *>(sf_smem_sfa)[lane_id] = sfa_vec;

            // Fill SFB SMEM buffer
            uint32_t sfb_fill = sfb_val | (sfb_val << 8) |
                                (sfb_val << 16) | (sfb_val << 24);
            uint4 sfb_vec = {sfb_fill, sfb_fill, sfb_fill, sfb_fill};
            reinterpret_cast<uint4 *>(sf_smem_sfb)[lane_id] = sfb_vec;

            __syncwarp();

            // Issue UTCCP: SMEM → TMEM with broadcast to all 4 subpartitions
            if (lane_id == 0) {
              // Construct SmemDescriptor for SFA buffer
              uint32_t sfa_smem_addr = static_cast<uint32_t>(
                  __cvta_generic_to_shared(sf_smem_sfa));
              uint64_t sfa_desc = 0;
              sfa_desc |= (uint64_t)((sfa_smem_addr >> 4) & 0x3FFF); // start_address [0:14)
              sfa_desc |= (uint64_t)(8) << 32;  // stride_byte_offset=8 [32:46)
              sfa_desc |= (uint64_t)(1) << 46;  // version=1 [46:48)

              asm volatile(
                  "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                  :: "r"(tmem_sfa_addr), "l"(sfa_desc));

              // Construct SmemDescriptor for SFB buffer
              uint32_t sfb_smem_addr = static_cast<uint32_t>(
                  __cvta_generic_to_shared(sf_smem_sfb));
              uint64_t sfb_desc = 0;
              sfb_desc |= (uint64_t)((sfb_smem_addr >> 4) & 0x3FFF);
              sfb_desc |= (uint64_t)(8) << 32;
              sfb_desc |= (uint64_t)(1) << 46;

              asm volatile(
                  "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                  :: "r"(tmem_sfb_addr), "l"(sfb_desc));
            }
          }

          // Issue 4 MMA k_blocks with same SF (granularity 128 = bK)
          for (int k_block = 0; k_block < cute::size<2>(tCrA);
               ++k_block) {
            cute::gemm(
                tiled_mma.with(tiled_mma.accumulate_, tCtSFA, tCtSFB),
                tCrA(cute::_, cute::_, k_block, smem_rd_buffer),
                tCrB(cute::_, cute::_, k_block, smem_rd_buffer),
                tCtAcc_Slice);
            tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
          }

          cutlass::arch::umma_arrive(
              &shared_storage.ab_empty_mbar_ptr[smem_rd_buffer]);

          if (mma_rd_k_tile_next < k_tile_count) {
            peek_ab_full_status = kernel::try_wait_barrier(
                shared_storage.ab_full_mbar_ptr[smem_rd_buffer_next],
                mma_rd_ab_full_phase_next);
          }

          mma_rd_k_tile = mma_rd_k_tile_next;
          smem_rd_buffer = smem_rd_buffer_next;
          mma_rd_ab_full_phase = mma_rd_ab_full_phase_next;
        } // end for k_tile

        cutlass::arch::umma_arrive(
            &shared_storage.acc_full_mbar_ptr[acc_buf_idx]);
        num_tiles_executed++;
      } // end for n_tile
    }

  }
  // ===============================================================
  // Warps 0-3: Epilogue — TMEM → registers → BF16 → smem → TMA
  // ===============================================================
  else if (warp_idx < 4) {
    if (warp_idx == 0) {
      tmem_allocator.allocate(num_tmem_columns,
                              &shared_storage.tmem_base_ptr);
    }
    tmem_allocation_result_barrier.arrive_and_wait();
    tCtAcc.data() = shared_storage.tmem_base_ptr;

    using AccType = typename decltype(tCtAcc)::value_type;
    using TypeC = OutType;

    cutlass::NumericConverter<TypeC, AccType> converter;

    cute::TiledCopy tiled_copy_t2r =
        cute::make_tmem_copy(cute::SM100_TMEM_LOAD_32dp32b1x{},
                             tCtAcc(cute::_, cute::_, cute::_, 0));
    cute::ThrCopy thr_copy_t2r = tiled_copy_t2r.get_slice(threadIdx.x);
    cute::Tensor tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc);

    cute::Tensor tCgC_fake = cute::make_tensor<TypeC>(
        cute::shape(tCtAcc(cute::_, cute::_, cute::_, 0)));
    cute::Tensor tTR_rAcc_fake = thr_copy_t2r.partition_D(tCgC_fake);
    cute::Tensor tTR_rAcc =
        cute::make_tensor<AccType>(cute::shape(tTR_rAcc_fake));

    int num_tiles_executed = 0;
    for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
      for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {
        int acc_buf_idx = num_tiles_executed % NUM_ACC_STAGE;
        int acc_full_phase = num_tiles_executed / NUM_ACC_STAGE % 2;
        int c_smem_wr_buffer_idx = num_tiles_executed % NUM_C_STAGE;

        cute::Tensor tCrC = cute::make_tensor<TypeC>(
            cute::shape(tTR_rAcc(0, cute::_, 0, 0)));

        mm_output_smem.set_ptr(
            mm_output +
            c_smem_wr_buffer_idx * MMA_N * OUTPUT_ATOM_SIZE);

        cute::wait_barrier(
            shared_storage.acc_full_mbar_ptr[acc_buf_idx],
            acc_full_phase);

        // T2R copy
        cute::copy(tiled_copy_t2r,
                   tTR_tAcc(cute::_, cute::_, cute::_, cute::_,
                            acc_buf_idx),
                   tTR_rAcc);

        epilogue_wg_barrier.arrive_and_wait();
        if (cute::elect_one_sync()) {
          cute::arrive_barrier(
              shared_storage.acc_empty_mbar_ptr[acc_buf_idx]);
        }

        // Convert FP32 accumulator to BF16
        CUTE_UNROLL
        for (int i = 0; i < tCrC.size(); i++) {
          tCrC[i] = converter(tTR_rAcc[i]);
        }

        // R2S copy
        cute::Tensor sC_epi_slice =
            cute::flatten(sC_epi(cute::_, 0, 0, c_smem_wr_buffer_idx));
        cute::copy(tCrC, sC_epi_slice(cute::_, threadIdx.x));

        // S2G TMA
        cute::tma_store_fence();
        epilogue_wg_barrier.arrive_and_wait();

        if (warp_idx == 0 && cute::elect_one_sync()) {
          tma_out.tma_store_async(
              mm_output_smem.base_ptr,
              {m_tile * OUTPUT_ATOM_SIZE, n_tile * MMA_N});
          cute::tma_store_arrive();
          cute::tma_store_wait<NUM_C_STAGE - 1>();
        }

        num_tiles_executed++;
      }
    }
    // Wait for all TMA stores
    if (warp_idx == 0 && cute::elect_one_sync()) {
      cute::tma_store_wait<0>();
    }
  }
  __syncthreads();

  if (warp_idx == 0) {
    tmem_allocator.free(shared_storage.tmem_base_ptr, num_tmem_columns);
  }
} // end fp8_linear_sm100_mpk_task_impl

} // namespace kernel
