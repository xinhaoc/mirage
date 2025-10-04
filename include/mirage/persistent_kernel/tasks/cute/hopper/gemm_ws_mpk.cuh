/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 #pragma once

 #include "cutlass/arch/mma_sm90.h"
 #include "cutlass/arch/reg_reconfig.h"
 #include "cutlass/cutlass.h"
 #include "cutlass/epilogue/collective/detail.hpp"
 #include "cutlass/fast_math.h"
 #include "cutlass/gemm/dispatch_policy.hpp"
 #include "cutlass/gemm/gemm.h"
 #include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
 #include "cutlass/kernel_hardware_info.hpp"
 #include "cutlass/pipeline/pipeline.hpp"
 #include "cutlass/trace.h"
 
 #include "cutlass/conv/detail.hpp"
 
 #include "cute/arch/cluster_sm90.hpp"
 #include "cute/tensor.hpp"
 
 #include "cutlass/arch/grid_dependency_control.h"
 #include "cutlass/arch/memory.h"

 // MPK Settings
 #include "../../hopper/tma.cuh"
 #include "../../hopper/smem_layout_tma.cuh"
 #include "../../hopper/utils.cuh"

 
 namespace kernel {
 
 using namespace cute;
 
 template <class CollectiveMainloop, class CollectiveEpilogue, bool TMAOnHost, typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE, typename TMA_A, typename TMA_B, typename TMA_OUT, typename TMA_RESIDUAL = void>
 CUTLASS_DEVICE void linear_cutlass_ws_hopper(
     typename CollectiveMainloop::template Params<TMAOnHost> const &mainloop_params,
     typename CollectiveEpilogue::Params const &epilogue_params,
     const TMA_A &tma_a,
     const TMA_B &tma_b,
     const TMA_OUT &tma_out,
     const TMA_RESIDUAL *tma_residual = nullptr) {
  //  if (threadIdx.x == 0) {
  //   printf("blockIdx.x: %d, threadIdx.x: %d, Entering linear_cutlass_ws_hopper, batch_size: %d, output_size: %d, reduction_size: %d\n", blockIdx.x, threadIdx.x, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE);
  //  }
 
   struct SharedStorage {
     // Mainloop and epilogue don't use smem concurrently since kernel is
     // non-persistent, so we can use a union
     union TensorStorage {
       using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
       using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
 
       MainloopTensorStorage mainloop;
       EpilogueTensorStorage epilogue;
     } tensors;
 
     struct PipelineStorage : cute::aligned_struct<16, _1> {
       using MainloopPipelineStorage =
           typename CollectiveMainloop::PipelineStorage;
       using EpiLoadPipelineStorage =
           typename CollectiveEpilogue::PipelineStorage;
 
       alignas(16) MainloopPipelineStorage mainloop;
       alignas(16) EpiLoadPipelineStorage epi_load;
     } pipelines;
   };
 
   enum class WarpGroupRole {
     Producer = 0,
     Consumer = 1,
   };
   enum class ProducerWarpRole {
     MainloopEpilogue = 0,
     Warp1 = 1,
     Warp2 = 2,
     Warp3 = 3
   };


  // auto mma_shape_A = cute::partition_shape_A(tiled_mma, cute::make_shape(cute::Int<MMA_M>{}, cute::size<2>(mma_tiler), cute::Int<NUM_AB_STAGE>{}));
  // // Pre-partitioned Tile Shape (MmaTile_N, MmaTile_K) to post-partitioned (MmaB, NumMma_N, NumMma_K)
  // auto mma_shape_B = cute::partition_shape_B(tiled_mma, cute::make_shape(cute::Int<MMA_N>{}, cute::size<2>(mma_tiler), cute::Int<NUM_AB_STAGE>{}));
  // // Pre-partitioned Tile Shape (MmaTile_N, MmaTile_M) to post-partitioned (MmaC, NumMma_N, NumMma_K)
  // auto mma_shape_C = cute::make_shape(cute::make_shape(cute::Int<MMA_N>{}, cute::Int<MMA_M>{}), cute::Int<1>{}, cute::Int<1>{}, cute::Int<NUM_C_STAGE>{});

  // // Print and inspect mma_shape_A, and mma_shape_B for this example.
  // // if (cute::thread0()) {
  // //     cute::print("mma_shape_A:\t"); cute::print(mma_shape_A); cute::print("\n");  // mma_shape_A:  ((_128,_16),_1,_4,_8)
  // //     cute::print("mma_shape_B:\t"); cute::print(mma_shape_B); cute::print("\n");  // mma_shape_B:  ((_32,_16),_1,_4,_8)
  // //     cute::print("mma_shape_C:\t"); cute::print(mma_shape_C); cute::print("\n");  // mma_shape_C:  ((_32,_128),_1,_1,_4)
  // // } __syncthreads();

  // auto sA_layout = cute::UMMA::tile_to_mma_shape(cute::UMMA::Layout_K_SW128_Atom<T_>{}, mma_shape_A);
  // auto sB_layout = cute::UMMA::tile_to_mma_shape(cute::UMMA::Layout_K_SW128_Atom<T_>{}, mma_shape_B);


   extern __shared__ char smem[];
   uintptr_t aligned_smem = (reinterpret_cast<uintptr_t>(smem) + 1023) / 1024 * 1024;

  //  if (threadIdx.x == 0) {
  //   printf("blockIdx.x: %d, threadIdx.x: %d, Aligning smem: %llu\n", blockIdx.x, threadIdx.x, aligned_smem);
  //  }
 
   using ClusterShape = typename CollectiveMainloop::ClusterShape;
   using TiledMma = typename CollectiveMainloop::TiledMma;
   using TileShape = typename CollectiveMainloop::TileShape;
   using SmemLayoutA = typename CollectiveMainloop::SmemLayoutA; // (BLK_M,BLK_K,PIPE)
   using SmemLayoutB = typename CollectiveMainloop::SmemLayoutB; // (BLK_N,BLK_K,PIPE)
   // using PipelineState = typename CollectiveMainloop::PipelineState;
 
   // Kernel level shared memory storage
   SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(aligned_smem);

   auto mma_tiler = cute::make_shape(OUTPUT_SIZE, BATCH_SIZE, REDUCTION_SIZE);

   cutlass::bfloat16_t *shared_weight = shared_storage.tensors.mainloop.smem_A.begin();
   cutlass::bfloat16_t *shared_input = shared_storage.tensors.mainloop.smem_B.begin();
  //  T_ *mm_output = shared_storage.tensors.epilogue.begin();
 
  //  Barrier *ab_full_mbar_ptr = reinterpret_cast<Barrier *>(shared_storage.ab_full_mbar_ptr);
 
  constexpr int INPUT_TMA_TILE_SIZE = 64;
  constexpr int WEIGHT_TMA_TILE_SIZE = INPUT_TMA_TILE_SIZE;
  constexpr int OUTPUT_TMA_TILE_SIZE = OUTPUT_SIZE < 64 ? OUTPUT_SIZE : 64;
  constexpr int OUTPUT_ATOM_SIZE = 64; // this is padded if OUTPUT_SIZE < 64
  constexpr bool HAS_RESIDUAL = !std::is_void<TMA_RESIDUAL>::value;
  constexpr int B = 3, M = 3, S = 3;
  constexpr int TILE_SIZE = 64;

  // NOTE(Yu): Assume batch size is smaller than 16, and padding the batch size
  // to 16
  static_assert(BATCH_SIZE <= 16,
                "Batch size must be smaller or equal to 16 in swapAB");
  constexpr int SMEM_M_SIZE = 16;
    using InputSmem = smem_tma<cutlass::bfloat16_t,
                             B,
                             M,
                             S,
                             SMEM_M_SIZE,
                             INPUT_TMA_TILE_SIZE,
                             TILE_SIZE / INPUT_TMA_TILE_SIZE>;
   InputSmem input_smem(shared_input);
 
   using WeightSmem = smem_tma<cutlass::bfloat16_t,
                               B,
                               M,
                               S,
                               OUTPUT_ATOM_SIZE,
                               WEIGHT_TMA_TILE_SIZE,
                               TILE_SIZE / WEIGHT_TMA_TILE_SIZE>;
   WeightSmem input_weight_smem(shared_weight);
 
  //  using ResidualSmem = smem_tma<cutlass::bfloat16_t,
  //                                B,
  //                                M,
  //                                S,
  //                                SMEM_M_SIZE,
  //                                OUTPUT_TMA_TILE_SIZE,
  //                                OUTPUT_ATOM_SIZE / OUTPUT_TMA_TILE_SIZE>;
  //  ResidualSmem residual_smem(shared_residual);
 
    // cute::Tensor mA = cute::make_coord_tensor(cute::make_layout(cute::make_shape(OUTPUT_SIZE, REDUCTION_SIZE), cute::make_stride(cute::E<1>{}, cute::E<0>{}))); // ArithTuple(_0,_0) o (output_size,reduction_size):(_1@1,_1@0)
    // cute::Tensor mB = cute::make_coord_tensor(cute::make_layout(cute::make_shape(BATCH_SIZE, REDUCTION_SIZE), cute::make_stride(cute::E<1>{}, cute::E<0>{}))); // ArithTuple(_0,_0) o (batch_size,reduction_size):(_1@1,_1@0)
    // cute::Tensor mC = cute::make_coord_tensor(cute::make_layout(cute::make_shape(BATCH_SIZE, OUTPUT_SIZE), cute::make_stride(cute::E<1>{}, cute::E<0>{}))); // ArithTuple(_0,_0) o (batch_size,output_size):(_1@1,_1@0)
 
 
   int thread_idx = int(threadIdx.x);
   int lane_idx = cutlass::canonical_lane_idx();
   int warp_idx = cutlass::canonical_warp_idx_sync();
   int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
   int warp_group_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
   auto warp_group_role = WarpGroupRole(cutlass::canonical_warp_group_idx());
   auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
   int lane_predicate = cute::elect_one_sync();
   uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
 
   // Issue Tma Descriptor Prefetch from a single thread
   if ((warp_idx == 0) && lane_predicate) {
    //  CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
    // if (threadIdx.x == 0) {
    //   printf("blockIdx.x: %d, threadIdx.x: %d, Prefetching tma descriptors\n", blockIdx.x, threadIdx.x);
    // }

      // prefetch_tma_descriptor(tma_a.desc_ptr);
      // prefetch_tma_descriptor(tma_b.desc_ptr);
      // prefetch_tma_descriptor(tma_out.desc_ptr);
      // if constexpr (HAS_RESIDUAL) {
      //   prefetch_tma_descriptor(tma_residual->desc_ptr);
      // }

      // if (threadIdx.x == 0) {
      //   printf("blockIdx.x: %d, threadIdx.x: %d, Prefetched tma descriptors\n", blockIdx.x, threadIdx.x);
      // }
    //  NOTE(Yu): prefetch epilogue tma descriptor if needed
     // CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
   }
 
   // Mainloop Load pipeline
   using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
   typename MainloopPipeline::Params mainloop_pipeline_params;
   if (warp_group_role == WarpGroupRole::Producer &&
       producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
     mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
   }
   if (warp_group_role == WarpGroupRole::Consumer) {
     mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
   }
 
   mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
   mainloop_pipeline_params.num_consumers = cutlass::NumThreadsPerWarpGroup;
   mainloop_pipeline_params.transaction_bytes =
       mainloop_params.tma_transaction_bytes;
   MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop,
                                      mainloop_pipeline_params,
                                      ClusterShape{});
 
   // Epilogue Load pipeline
   using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
   typename EpiLoadPipeline::Params epi_load_pipeline_params;
   if (warp_group_role == WarpGroupRole::Producer &&
       producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
     epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
   }
   if (warp_group_role == WarpGroupRole::Consumer) {
     epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
   }
   epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
   epi_load_pipeline_params.producer_arv_count = cutlass::NumThreadsPerWarp;
   epi_load_pipeline_params.consumer_arv_count = cutlass::NumThreadsPerWarpGroup;
   if constexpr (CollectiveEpilogue::RequiresTransactionBytes) {
     epi_load_pipeline_params.transaction_bytes =
         epilogue_params.tma_transaction_bytes;
   }
   EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load,
                                     epi_load_pipeline_params);
   // Epilogue Store pipeline
   using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
   typename EpiStorePipeline::Params epi_store_pipeline_params;
   epi_store_pipeline_params.always_wait = true;
   EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);
   // Initialize starting pipeline states for the collectives
   // Epilogue store pipe is producer-only (consumer is TMA unit, waits via
   // scoreboarding)
   typename CollectiveMainloop::PipelineState smem_pipe_read;
   typename CollectiveMainloop::PipelineState smem_pipe_release = smem_pipe_read;
   typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;
 
   // For the DMA Load (producer) we start with an opposite phase
   // i.e., we skip all waits since we know that the buffer is indeed empty
   cutlass::PipelineState mainloop_pipe_producer_state =
       cutlass::make_producer_start_state<MainloopPipeline>();
   cutlass::PipelineState epi_load_pipe_producer_state =
       cutlass::make_producer_start_state<EpiLoadPipeline>();
   cutlass::PipelineState epi_store_pipe_producer_state =
       cutlass::make_producer_start_state<EpiStorePipeline>();
   auto blk_shape = TileShape{}; // (BLK_M,BLK_N,BLK_K)
   TiledMma tiled_mma;
 
   auto problem_shape_mnkl =
       append<4>(mainloop_params.problem_shape, cute::Int<1>{});
 
   CollectiveMainloop collective_mainloop;
   CollectiveEpilogue collective_epilogue(epilogue_params);
 
//    auto load_inputs =
//        collective_mainloop.load_init(problem_shape_MNKL, mainloop_params);
  //   {
  //       using X = Underscore;
  //       auto [M, N, K, L] = problem_shape_MNKL;
  //       Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(
  //           make_shape(M, K, L)); // (m,k,l)
  //       Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(
  //           make_shape(N, K, L)); // (n,k,l)
  //       Tensor gA_mkl = local_tile(mA_mkl,
  //                                  TileShape{},
  //                                  make_coord(_, _, _),
  //                                  Step<_1, X, _1>{}); // (BLK_M,BLK_K,m,k,l)
  //       Tensor gB_nkl = local_tile(mB_nkl,
  //                                  TileShape{},
  //                                  make_coord(_, _, _),
  //                                  Step<X, _1, _1>{}); // (BLK_N,BLK_K,n,k,l)
    
  //       return cute::make_tuple(gA_mkl, gB_nkl);   
  //   }
  //  static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2,
  //                "Output of load_init must have at least two elements (A, B)");
 
  //  // Extract out partitioned A and B.
  //  Tensor gA_mkl = get<0>(load_inputs);
  //  Tensor gB_nkl = get<1>(load_inputs);

 
   // Compute m_coord, n_coord, and l_coord with their post-tiled shapes
  //  auto m_coord = idx2crd(int(blockIdx.x), shape<2>(gA_mkl));
  //  auto n_coord = idx2crd(int(blockIdx.y), shape<2>(gB_nkl));
   // handles the difference between the rank of Tensor returned by load_input in
   // case they do not have a batch mode auto l_coord = [&] (auto const& gB_nkl_)
   // {
   //   // gB_nkl needs to be passed into the lambda because C++17
   //   // does not permit lambda capture of structured bindings.
   //   if constexpr (not IsConvProblemShape) {
   //     // This needs to be inside an `if constexpr`,
   //     // because shape<4>(gB_nkl) is not well-formed otherwise.
   //     return idx2crd(int(blockIdx.z), shape<4>(gB_nkl_));
   //   }
   //   else {
   //     return Int<0>{};
   //   }
   // } (gB_nkl);
 
  //  if (threadIdx.x == 0) {
  //   printf("blockIdx.x: %d, threadIdx.x: %d, Making blk_coord\n", blockIdx.x, threadIdx.x);
  //  }
   auto blk_coord = make_coord(Int<0>{}, Int<0>{}, _, Int<0>{});
 
   // Get pipeline iterators and increments from tensor shapes
  //  auto k_tile_iter = cute::make_coord_iterator(shape<3>(gA_mkl));
  //  auto k_tile_count = size<3>(gA_mkl);
  constexpr int NUM_ITERS_K = (REDUCTION_SIZE + TILE_SIZE - 1) / TILE_SIZE;
  auto k_tile_count = NUM_ITERS_K;
 
   // Wait for all thread blocks in the Cluster
   __syncthreads();
 
   if (warp_group_role == WarpGroupRole::Producer) {
     if (producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
       // Ensure that the prefetched kernel does not touch
       // unflushed global memory prior to this instruction
       cutlass::arch::wait_on_dependent_grids();
    //    collective_mainloop.load(mainloop_params,
    //                             mainloop_pipeline,
    //                             mainloop_pipe_producer_state,
    //                             load_inputs,
    //                             blk_coord,
    //                             k_tile_iter,
    //                             k_tile_count,
    //                             lane_idx,
    //                             block_rank_in_cluster,
    //                             shared_storage.tensors.mainloop);
       {
            int lane_predicate = cute::elect_one_sync();

            if (lane_predicate) {
            // Tensor sA = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_A.data()),
            //                         SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
            // Tensor sB = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_B.data()),
            //                         SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

            // Tensor gA_mkl = get<0>(load_inputs);
            // Tensor gB_nkl = get<1>(load_inputs);
            // CUTE_STATIC_ASSERT_V(size<2>(gB_nkl) == 0); //
            // int m_tile_count = size<2>(gA_mkl);
            // int n_tile_count = size<2>(gB_nkl);

            // auto block_tma_a = mainloop_params.tma_load_a.get_slice(0);
            // auto block_tma_b = mainloop_params.tma_load_b.get_slice(0);

            // Partition the inputs based on the current block coordinates.
            // auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

            // Tensor gA = gA_mkl(_, _, m_coord, _, l_coord); // (BLK_M,BLK_K,k)
            // Tensor gB = gB_nkl(_, _, n_coord, _, l_coord); // (BLK_N,BLK_K,k)
            // auto gA = cute::local_tile(mA, mma_tiler, mma_coord, cute::Step<cute::_1, cute::X, cute::_1>{});  // (BLK_M,BLK_K, m,k)
            // auto gB = cute::local_tile(mB, mma_tiler, mma_coord, cute::Step<cute::X, cute::_1, cute::_1>{});  // (BLK_N,BLK_K, n,k)


            // Applies the mapping from block_tma_a
            // Tensor tAgA = block_tma_a.partition_S(gA); // (TMA,TMA_M,TMA_K,k)
            // Tensor tAsA = block_tma_a.partition_D(sA); // (TMA,TMA_M,TMA_K,PIPE)

            // Tensor tBgB = block_tma_b.partition_S(gB); // (TMA,TMA_N,TMA_K,k)
            // Tensor tBsB = block_tma_b.partition_D(sB); // (TMA,TMA_N,TMA_K,PIPE)

            // uint16_t mcast_mask_a = 0;
            // uint16_t mcast_mask_b = 0;

            // Mainloop
            CUTLASS_PRAGMA_NO_UNROLL
            for (int k_iter = 0; k_iter < k_tile_count; k_iter++) {
              // if (threadIdx.x == 0) {
              //   printf("blockIdx.x: %d, threadIdx.x: %d, Producer loop k_iter: %d\n", blockIdx.x, threadIdx.x, k_iter);
              // }
                // LOCK smem_pipe_write for _writing_
                mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state);

                //
                // Copy gmem to smem for *k_iter
                //

                using BarrierType = typename MainloopPipeline::ProducerBarrierType;
                BarrierType *tma_barrier =
                    mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);

                int write_stage = mainloop_pipe_producer_state.index();

                  // wait(compute_done[slot], phase ^ 1);
                  int tma_coords_A[2] = {k_iter * TILE_SIZE,
                                        0 * OUTPUT_ATOM_SIZE};
                  int tma_coords_B[2] = {k_iter * TILE_SIZE, 0};
                  input_weight_smem.set_ptr(shared_weight +
                                            write_stage * OUTPUT_ATOM_SIZE * TILE_SIZE);
                  input_smem.set_ptr(shared_input + write_stage * SMEM_M_SIZE * TILE_SIZE);
                  tma_a.tma_cp_async(
                      *tma_barrier, input_weight_smem(0, 0), tma_coords_A);
                  tma_b.tma_cp_async(
                      *tma_barrier, input_smem(0, 0), tma_coords_B);
                  // if (threadIdx.x == 0) {
                  //   printf("blockIdx.x: %d, threadIdx.x: %d, Producer loop k_iter: %d, Tma cp async\n", blockIdx.x, threadIdx.x, k_iter);
                  // }
                // printf("producer really start\n");
                // copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a),
                //     tAgA(_, _, _, *k_tile_iter),
                //     tAsA(_, _, _, write_stage));
                // copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b),
                //     tBgB(_, _, _, *k_tile_iter),
                //     tBsB(_, _, _, write_stage));

                // ++k_iter;

                // Advance smem_pipe_write
                ++mainloop_pipe_producer_state;

                // if (threadIdx.x == 0) {
                //   printf("blockIdx.x: %d, threadIdx.x: %d, Producer loop k_iter: %d, Advance smem_pipe_write\n", blockIdx.x, threadIdx.x, k_iter);
                // }

            }
        }





       }
       // Update starting mainloop pipeline state for the pipeline drain
      //  mainloop_pipe_producer_state.advance(k_tile_count);
       // Make sure mainloop consumer has been waited upon before issuing
       // epilogue load
    //    collective_mainloop.load_tail(mainloop_pipeline,
    //                                  mainloop_pipe_producer_state);
        {
            int lane_predicate = cute::elect_one_sync();

            // Issue the epilogue waits
            if (lane_predicate) {
              //  pipeline.producer_tail(smem_pipe_write);
              mainloop_pipeline.producer_tail(mainloop_pipe_producer_state);
            }
        }
     }
   } else if (warp_group_role == WarpGroupRole::Consumer) {
    // if (threadIdx.x == 128) {
    //   printf("blockIdx.x: %d, threadIdx.x: %d, Consumer loop\n", blockIdx.x, threadIdx.x);
    // }
     Tensor accum = partition_fragment_C(
         tiled_mma, take<0, 2>(blk_shape)); // (MMA,MMA_M,MMA_N)
 
    //  collective_mainloop.mma(mainloop_pipeline,
    //                          mainloop_pipe_consumer_state,
    //                          accumulators,
    //                          k_tile_count,
    //                          warp_group_thread_idx,
    //                          shared_storage.tensors.mainloop,
    //                          mainloop_params);
    

        Tensor sA = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_A.data()),
        SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
        Tensor sB = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_B.data()),
            SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
        constexpr int MmaWarpGroups =
        size(TiledMma{}) / cutlass::NumThreadsPerWarpGroup;
        Layout warp_group_thread_layout = make_layout(
        Int<MmaWarpGroups>{}, Int<cutlass::NumThreadsPerWarpGroup>{});

        int warp_group_idx = __shfl_sync(
        0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        auto thread_mma =
        tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

        Tensor tCsA = thread_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
        Tensor tCsB = thread_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)

        // Allocate "fragments/descriptors"
        Tensor tCrA = thread_mma.make_fragment_A(tCsA); // (MMA,MMA_M,MMA_K,PIPE)
        Tensor tCrB = thread_mma.make_fragment_B(tCsB); // (MMA,MMA_N,MMA_K,PIPE)

        // CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum)); // M
        // CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum)); // N
        // CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));  // K
        // CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));  // PIPE
        // CUTE_STATIC_ASSERT_V(Int<KernelTraits::NUM_STAGES>{} ==
        // size<2>(sA)); // PIPE
        // CUTE_STATIC_ASSERT_V(Int<KernelTraits::NUM_STAGES>{} ==
        // size<2>(sB)); // PIPE

        //
        // PIPELINED MAIN LOOP
        //
        static_assert((0 <= CollectiveMainloop::K_PIPE_MMAS) &&
        (CollectiveMainloop::K_PIPE_MMAS < CollectiveMainloop::NUM_STAGES),
        "ERROR : Incorrect number of MMAs in flight");

        // We release buffers to producer warps(dma load) with some mmas in flight

        // Prologue GMMAs
        int prologue_mma_count = min(CollectiveMainloop::K_PIPE_MMAS, k_tile_count);
        assert(k_tile_count >= 1);
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        warpgroup_fence_operand(accum);
        {
        // WAIT on smem_pipe_read until its data are available (phase bit flips
        // from rdPhaseBit value)
        auto barrier_token = mainloop_pipeline.consumer_try_wait(smem_pipe_read);

        mainloop_pipeline.consumer_wait(smem_pipe_read, barrier_token);
        // if (threadIdx.x == 128) {
        //   printf("blockIdx.x: %d, threadIdx.x: %d, Consumer loop, after consumer wait\n", blockIdx.x, threadIdx.x);
        // }

        int read_stage = smem_pipe_read.index();
        warpgroup_arrive();
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        #if 0

        if (threadIdx.x == 128) {
          printf("prologue_mma_count: %d\n", prologue_mma_count);
          printf("BATCH_SIZE: %d, OUTPUT_SIZE: %d, REDUCTION_SIZE: %d\n", BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE);
          printf("tCsA: ");
          print(tCsA);
          // print_tensor(tCsA);
          printf("tCsB: ");
          print(tCsB);
          printf("tCrA: ");
          print(tCrA);
          printf("tCrB: ");
          print(tCrB);
          printf("accum: ");
          print(accum);
          printf("tiled_mma: ");
          print(tiled_mma);
        }
        #endif
        // Unroll the K mode manually to set scale D to 1
        // if (threadIdx.x == 128) {
        //   printf("blockIdx.x: %d, threadIdx.x: %d, Consumer loop, before gemm\n", blockIdx.x, threadIdx.x);
        // }
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma,
        tCrA(_, _, k_block, read_stage),
        tCrB(_, _, k_block, read_stage),
        accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
        // if (threadIdx.x == 128) {
        //   printf("blockIdx.x: %d, threadIdx.x: %d, Consumer loop, after gemm\n", blockIdx.x, threadIdx.x);
        // }

        warpgroup_commit_batch();

        ++smem_pipe_read;
        }

        tiled_mma.accumulate_ = GMMA::ScaleOut::One;

        warpgroup_fence_operand(accum);
        // if (threadIdx.x == 128) {
        //   printf("blockIdx.x: %d, threadIdx.x: %d, Consumer loop, before second gemm\n", blockIdx.x, threadIdx.x);
        // }
        CUTLASS_PRAGMA_UNROLL
        for (int k_tile_prologue = prologue_mma_count - 1; k_tile_prologue > 0;
        --k_tile_prologue) {
        // WAIT on smem_pipe_read until its data are available (phase bit flips
        // from rdPhaseBit value)
        auto barrier_token = mainloop_pipeline.consumer_try_wait(smem_pipe_read);
        mainloop_pipeline.consumer_wait(smem_pipe_read, barrier_token);

        int read_stage = smem_pipe_read.index();
        warpgroup_arrive();
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma,
        tCrA(_, _, _, read_stage),
        tCrB(_, _, _, read_stage),
        accum);
        warpgroup_commit_batch();

        ++smem_pipe_read;
        }

        warpgroup_fence_operand(accum);
        // Mainloop GMMAs
        k_tile_count -= prologue_mma_count;

        CUTLASS_PRAGMA_NO_UNROLL
        for (; k_tile_count > 0; --k_tile_count) {
        // WAIT on smem_pipe_read until its data are available (phase bit flips
        // from rdPhaseBit value)
        auto barrier_token = mainloop_pipeline.consumer_try_wait(smem_pipe_read);
        mainloop_pipeline.consumer_wait(smem_pipe_read, barrier_token);

        //
        // Compute on k_tile
        //

        int read_stage = smem_pipe_read.index();
        warpgroup_fence_operand(accum);
        warpgroup_arrive();
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma,
        tCrA(_, _, _, read_stage),
        tCrB(_, _, _, read_stage),
        accum);
        warpgroup_commit_batch();

        /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to
        /// ensure smem_pipe_write is consumed
        warpgroup_wait<CollectiveMainloop::K_PIPE_MMAS>();
        warpgroup_fence_operand(accum);

        // UNLOCK smem_pipe_release, done _computing_ on it
        mainloop_pipeline.consumer_release(smem_pipe_release);

        // Advance smem_pipe_read and smem_pipe_release
        ++smem_pipe_read;
        ++smem_pipe_release;
        }

        warpgroup_fence_operand(accum);

    
 
     // Make sure the math instructions are done and free buffers before entering
     // the epilogue
    //  collective_mainloop.mma_tail(
    //      mainloop_pipeline, mainloop_pipe_consumer_state, k_tile_count);
    // {
        // Prologue GMMAs
        // int prologue_mma_count = min(CollectiveMainloop::K_PIPE_MMAS, k_tile_count);
        // k_tile_count -= prologue_mma_count;

        smem_pipe_release.advance(k_tile_count);

        // Wait on all GMMAs to complete
        warpgroup_wait<0>();

        for (int count = 0; count < prologue_mma_count; ++count) {
        mainloop_pipeline.consumer_release(smem_pipe_release); // UNLOCK smem_pipe_release,
                                                        // done _computing_ on it
        ++smem_pipe_release;
        }
    // }
 
     // Hint on an early release of global memory resources.
     // The timing of calling this function only influences performance,
     // not functional correctness.
     cutlass::arch::launch_dependent_grids();
 
     #if 0
     if (threadIdx.x == 128) {
       printf("problem_shape_MNKL: \n"); print(problem_shape_MNKL); printf("\n");
       printf("blk_shape: \n"); print(blk_shape); printf("\n");
       printf("blk_coord: \n"); print(blk_coord); printf("\n");
       printf("accumulators: \n"); print(accum); printf("\n");
       printf("tiled_mma: \n"); print(tiled_mma); printf("\n");
       printf("warp_group_thread_idx: \n"); print(warp_group_thread_idx); printf("\n");
     }
     #endif
     // Epilogue and write to gD
    //  auto [epi_load_pipe_consumer_state_next,
    //        epi_store_pipe_producer_state_next] =
    //      collective_epilogue.store(epi_load_pipeline,
    //                                epi_load_pipe_consumer_state,
    //                                epi_store_pipeline,
    //                                epi_store_pipe_producer_state,
    //                                problem_shape_MNKL,
    //                                blk_shape,
    //                                blk_coord,
    //                                accumulators,
    //                                tiled_mma,
    //                                warp_group_thread_idx,
    //                                shared_storage.tensors.epilogue);
    // {
        constexpr int BLK_M_RANK = cute::rank<0>(blk_shape);
        auto m_max_coord =
            unwrap(cute::transform(make_seq<BLK_M_RANK>{}, [&](auto i) {
            return get<0, i>(problem_shape_mnkl) -
                    get<0, i>(blk_shape) * get<0, i>(blk_coord);
            }));

        constexpr int BLK_N_RANK = cute::rank<1>(blk_shape);
        auto n_max_coord =
            unwrap(cute::transform(make_seq<BLK_N_RANK>{}, [&](auto i) {
            return get<1, i>(problem_shape_mnkl) -
                    get<1, i>(blk_shape) * get<1, i>(blk_coord);
            }));

        auto residue_mnk = make_tuple(m_max_coord, n_max_coord, Int<0>{});
        // store_op(problem_shape_mnkl,
        //     blk_shape,
        //     cta_coord_mnkl,
        //     accumulators,
        //     tiled_mma,
        //     residue_mnk,
        //     thread_idx,
        //     reinterpret_cast<char *>(&shared_tensors));

        // return cute::make_tuple(load_pipe_consumer_state,
        //                         store_pipe_producer_state);
        // start store op
        using X = Underscore;

        // Separate out problem shape for convenience
        auto M = get<0>(problem_shape_mnkl);
        auto N = get<1>(problem_shape_mnkl);
        auto L = get<3>(problem_shape_mnkl);

        // no transpose for epologue
        auto stride_c = epilogue_params.dC;
        auto stride_d = epilogue_params.dD;
        auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        if constexpr (::cutlass::gemm::kernel::detail::Has_SwapAB_v<CollectiveMainloop>) {
          // Represent the full output tensor
            Tensor mC_mnl = make_tensor(make_gmem_ptr<typename CollectiveEpilogue::DataTypeC>(epilogue_params.ptr_C),
            make_shape(M, N, L),
            stride_c); // (m,n,l)
            Tensor mD_mnl = make_tensor(
            make_gmem_ptr(epilogue_params.ptr_D), make_shape(M, N, L), stride_d); // (m,n,l)
            Tensor gC_mnl = local_tile(mC_mnl,
            blk_shape,
            make_coord(_, _, _),
            Step<_1, _1, X>{}); // (BLK_M,BLK_N,m,n,l)
            Tensor gD_mnl = local_tile(mD_mnl,
            blk_shape,
            make_coord(_, _, _),
            Step<_1, _1, X>{}); // (BLK_M,BLK_N,m,n,l)

            // Slice to get the tile this CTA is responsible for
            Tensor gC = gC_mnl(_, _, m_coord, n_coord, l_coord); // (BLK_M,BLK_N)
            Tensor gD = gD_mnl(_, _, m_coord, n_coord, l_coord); // (BLK_M,BLK_N)


            auto dD_T = cute::make_stride(cute::Int<1>{}, M, get<2>(epilogue_params.dD));   // transpose dD
            Tensor mD_mnl_T = cute::make_tensor(
            cute::make_gmem_ptr(epilogue_params.ptr_D),
            cute::make_shape(M, N, L),
            dD_T
            );
            Tensor gD_T = local_tile(mD_mnl_T, blk_shape, make_coord(_,_,_), Step<_1,_1,X>{})
                    (_,_, m_coord, n_coord, l_coord);   // (BLK_M, BLK_N)

            auto dC_T = cute::make_stride(cute::Int<1>{}, M, get<2>(epilogue_params.dC));
            Tensor mC_mnl_T = cute::make_tensor(cute::make_gmem_ptr<typename CollectiveEpilogue::DataTypeC>(epilogue_params.ptr_C),
                                                cute::make_shape(M, N, L), dC_T);
            Tensor gC_T = local_tile(mC_mnl_T, blk_shape, make_coord(_,_,_), Step<_1,_1,X>{})
                            (_,_, m_coord, n_coord, l_coord);

            // Partition source and destination tiles to match the accumulator
            // partitioning
            // Tensor tCgD = thr_mma.partition_C(gD); // (VEC,THR_M,THR_N)
            // Tensor tCgC = thr_mma.partition_C(gC); // (VEC,THR_M,THR_N)

            Tensor tCgD = thr_mma.partition_C(gD_T); // (VEC,THR_M,THR_N)
            Tensor tCgC = thr_mma.partition_C(gC_T); // (VEC,THR_M,THR_N)

            // OOB predication for tile quantization "residue"
            // Absolute coordinate tensors (dynamic)
            auto shape_MN = make_shape(M, N);
            Tensor mD_crd = make_identity_tensor(shape_MN); // (M,N)
            Tensor cD_mn = local_tile(mD_crd,
            take<0, 2>(blk_shape),
            make_coord(m_coord, n_coord)); // (BLK_M,BLK_N)
            Tensor tCcD_mn = thr_mma.partition_C(cD_mn); // (VEC,THR_M,THR_N)
            // Relative coordinate tensors (static)
            Tensor cD = cute::make_coord_tensor(cD_mn.layout()); // (BLK_M,BLK_N)
            Tensor tCcD =
            cute::make_coord_tensor(tCcD_mn.layout()); // (VEC,THR_M,THR_N)
            // Subtract the global "bottom right" corner from the local "top left"
            // corner to get the max relative coordinate
            auto residue_cD = shape_MN - cD_mn(_0{});     // (m,n)
            auto residue_tCcD = shape_MN - tCcD_mn(_0{}); // (m,n)

            // Fully OOB tile
            if (not elem_less(repeat_like(residue_cD, _0{}), residue_cD)) {
            return;
            }

            using FragCType = remove_cvref_t<decltype(tCgC(0))>;
            using FragDType = remove_cvref_t<decltype(tCgD(0))>;

            // source is needed
            // if (epilogue_op.is_source_needed()) {
            //   CUTLASS_PRAGMA_UNROLL
            //   for (int i = 0; i < size(accumulators); ++i) {
            //     FragCType fragC;
            //     bool pred = elem_less(tCcD(i), residue_tCcD);
            //     cutlass::arch::global_load<FragCType, sizeof(FragCType)>(
            //         fragC, &tCgC(i), pred);
            //     FragDType fragD = epilogue_op(accumulators(i), fragC);
            //     cutlass::arch::global_store<FragDType, sizeof(FragDType)>(
            //         fragD, &tCgD(i), pred);
            //   }
            // }
            // source is not needed, avoid load
            // else {

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(accum); ++i) {
            FragCType fragC;
            bool pred = elem_less(tCcD(i), residue_tCcD);
            cutlass::arch::global_load<FragCType, sizeof(FragCType)>(
            fragC, &tCgC(i), pred);
            FragDType fragD = collective_epilogue.epilogue_op(accum(i), fragC);

            cutlass::arch::global_store<FragDType, sizeof(FragDType)>(
            fragD, &tCgD(i), pred);
            // }
            }
        // }
    }
 
     // collective_epilogue.store_tail(epi_load_pipeline,
     //                                epi_load_pipe_consumer_state_next,
     //                                epi_store_pipeline,
     //                                epi_store_pipe_producer_state_next);
   }
  //  if (threadIdx.x == 0) {
  //   printf("blockIdx.x: %d, threadIdx.x: %d, Exiting linear_cutlass_ws_hopper\n", blockIdx.x, threadIdx.x);
  //  }
 }
 // };
 
 ///////////////////////////////////////////////////////////////////////////////
 
 } // namespace kernel
 