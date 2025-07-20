from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='runtime_kernel_hopper',
    ext_modules=[
        CUDAExtension(
            name='runtime_kernel_hopper',
            sources=[
                os.path.join(this_dir, 'matmul_demo.cu'),
            ],
            include_dirs=[
                os.path.join(this_dir, '../../../include/mirage/persistent_kernel/tasks'),
            ],
            extra_compile_args={
                'cxx': ['-DMIRAGE_GRACE_HOPPER'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_90a,code=sm_90a',
                    '-DMIRAGE_GRACE_HOPPER',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
