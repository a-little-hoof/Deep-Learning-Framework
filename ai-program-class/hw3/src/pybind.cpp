#include "tensor.h"
#include "layer.h"
#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<pybind11/numpy.h>
namespace py = pybind11;

Tensor* numpy_to_tensor(pybind11::array_t<float> array) {
    pybind11::buffer_info info = array.request();
    std::vector<int> shape(info.shape.begin(), info.shape.end());
    Tensor* tensor = new Tensor(shape, "cpu");
    tensor->data = (float*)info.ptr;
    return tensor;
}

PYBIND11_MODULE(mytorch, m) {
    py::class_<Tensor>(m, "Tensor")
    .def(py::init<const std::vector<int>&, const std::string&>())
    .def_readwrite("shape", &Tensor::shape)
    .def_readwrite("device", &Tensor::device)
    .def("cpu", &Tensor::cpu)
    .def("gpu", &Tensor::gpu)
    .def("get_size", &Tensor::get_size)
    .def("print", &Tensor::print)
    .def("fill_", &Tensor::fill_)
    .def("copy", &Tensor::copy)
    .def_property_readonly("data", [](Tensor &t) {
        std::vector<ssize_t> shape(t.shape.begin(), t.shape.end());
        std::vector<ssize_t> strides(shape.size());
        ssize_t stride = sizeof(float);
        for (ssize_t i = shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return py::array_t<float>(
            t.shape,   // shape
            strides, // strides
            t.data,  // data pointer
            py::cast(t) // base object
        );
    });

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

    // ReLU
    m.def("relu_forward", &relu_forward, "ReLU forward");
    m.def("relu_backward", &relu_backward, "ReLU backward");

    // Sigmoid
    m.def("sigmoid_forward", &sigmoid_forward, "Sigmoid forward");
    m.def("sigmoid_backward", &sigmoid_backward, "Sigmoid backward");

    m.def("numpy_to_tensor", &numpy_to_tensor, "Convert numpy array to Tensor");
}