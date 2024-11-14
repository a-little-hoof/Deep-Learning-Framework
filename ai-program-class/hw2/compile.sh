export PATH=$HOME/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cuda-12.0/lib64:$LD_LIBRARY_PATH

nvcc -std=c++11 -o layer tensor.cu layer.cu main.cpp -I. -lcublas -lcurand