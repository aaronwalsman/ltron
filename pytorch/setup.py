from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='remap_sum',
        ext_modules=[cpp_extension.CppExtension(
            'remap_sum', ['remap_sum.cpp'])],
        cmdclass={'build_ext' : cpp_extension.BuildExtension}
)
