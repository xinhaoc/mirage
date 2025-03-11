// runtime.h - Runtime for Program Generated by Mirage
#pragma once

#include <vector>

#include <cuda_runtime_api.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

// The following two functions will be generated by the transpiler
static void _init();
static void _execute_mugraph(std::vector<void const *> input_tensors,
                             std::vector<void *> output_tensors,
                             void *buf,
                             int rank = 0);

// Runtime libraries
#include "config.h"
#include "kernel/element_binary.h"
#include "kernel/element_unary.h"
#include "kernel/matmul.h"
#include "kernel/reduction.h"
#include "kernel/communication.h"
//#include "nvshmem.h"
#include "threadblock/threadblock.h"
#ifdef USE_NVSHMEM
#include "threadblock/comm_executor.h"
#endif
#include "utils.h"

// Entrypoint for C/C++
extern "C" void execute_mugraph(std::vector<void const *> input_tensors,
                                std::vector<void *> output_tensors,
                                void *buf,
                                int rank = 0) {
  static bool inited = false;
  if (!inited) {
    _init();
  }
  _execute_mugraph(input_tensors, output_tensors, buf, rank);
}

// A wrappr around `execute_mugraph` which uses C arrays instead of vectors
// Entrypoint for Python
void execute_mugraph_wrapper(void const *input_tensors[],
                             size_t num_input_tensors,
                             void *output_tensors[],
                             size_t num_output_tensors,
                             void *buf,
                             int rank = 0) {
  std::vector<void const *> input_tensors_vec(
      input_tensors, input_tensors + num_input_tensors);
  std::vector<void *> output_tensors_vec(output_tensors,
                                         output_tensors + num_output_tensors);
  execute_mugraph(input_tensors_vec, output_tensors_vec, buf, rank);
}
