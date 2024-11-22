#include "tensor.h"
#include "layer.h"
#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(mytorch, m) {
    py::class_<Tensor>(m, "Tensor")
    .def(py::init<const std::vector<int>&, const std::string&>())
    .def_readwrite("shape", &Tensor::shape)
    .def_readwrite("device", &Tensor::device)
    .def_readwrite("data", &Tensor::data)
    .def("cpu", &Tensor::cpu)
    .def("gpu", &Tensor::gpu)
    .def("get_size", &Tensor::get_size)
    .def("print", &Tensor::print)
    .def("fill_", &Tensor::fill_);

    // 绑定 layer 函数
    m.def("matrix_init", &matrix_init, "Initialize matrix in Tensor");

    // Fully connected layer functions
    m.def("fc_forward", &fc_forward, "Fully connected forward");
    m.def("fc_backward", &fc_backward, "Fully connected backward");

    // Convolutional layer functions
    m.def("conv_forward", &conv_forward, "Convolutional forward");
    m.def("conv_backward", &conv_backward, "Convolutional backward");

    // Maxpool layer functions
    m.def("maxpool_forward", &maxpool_forward, "Maxpool forward");
    m.def("maxpool_backward", &maxpool_backward, "Maxpool backward");

    // Softmax
    m.def("softmax_forward", &softmax_forward, "Softmax forward");

    // Cross-entropy loss
    m.def("cross_entropy_forward", &cross_entropy_forward, "Cross entropy loss forward");
    m.def("cross_entropy_with_softmax_backward", &cross_entropy_with_softmax_backward,
          "Cross entropy loss backward with softmax");
}