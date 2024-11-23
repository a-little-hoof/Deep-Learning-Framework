import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
__version__ = '0.0.1'
sources = ['src/pybind.cpp','src/tensor.cu', 'src/layer.cu']
include_dirs = ['src']
setup(
    name='mytorch',
    version=__version__,
    author='YifeiWang',
    author_email='wyf181030@stu.pku.edu.cn',
    packages=find_packages(),
    zip_safe=False,
    install_requires=['torch'],
    python_requires='>=3.8',
    license='MIT',
    ext_modules=[
        CUDAExtension(
        name='mytorch',
        sources=sources)
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
    ],
)