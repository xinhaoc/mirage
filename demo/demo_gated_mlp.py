import mirage as mi
import torch

if __name__ == "__main__":
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(8, 4096), dtype=mi.float16)
    W1 = graph.new_input(dims=(4096, 4096), dtype=mi.float16)
    W2 = graph.new_input(dims=(4096, 4096), dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(grid_dim=(64,1,1), block_dim=(128,1,1), forloop_range=64, reduction_dimx=64)
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tW1 = tb_graph.new_input(dtensor=W1, input_map=(1, -1, -1), forloop_dim=0)
    tW2 = tb_graph.new_input(dtensor=W2, input_map=(1, -1, -1), forloop_dim=0)
    tD1 = tb_graph.matmul(tX, tW1)
    tD2 = tb_graph.matmul(tX, tW2)
    tA1 = tb_graph.forloop_accum(tD1)
    tA2 = tb_graph.forloop_accum(tD2)
    tS = tb_graph.silu(tA1)
    tO = tb_graph.mul(tS, tA2)
    tb_graph.new_output(stensor=tO, output_map=(1, -1, -1))
    O = graph.customized([X, W1, W2], tb_graph)
    
    input_tensors = [
        torch.randn(8, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, 4096, dtype=torch.float16, device='cuda:0')
    ]

    input_strides = [tensor.stride() for tensor in input_tensors]
    p = mi.generate_cuda_program(graph.cygraph, target_cc=86, input_strides=input_strides, output_tensors=O)
    print(p["code"])
    #outputs = graph(inputs=input_tensors, outputs=[O])