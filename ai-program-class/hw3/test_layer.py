import unittest
import torch
import numpy as np
import mytorch
import copy

def MyTensorToTorchTensor(my_tensor):
    my_tensor.cpu()
    my_tensor_data = np.array(my_tensor.data[:my_tensor.get_size()])
    torch_tensor_data = torch.tensor(my_tensor_data)
    return torch_tensor_data

class TestLayer(unittest.TestCase):

    def test_matrix_init(self):
        shape = [2, 3]
        device = "GPU"

        # Initialize custom Tensor
        my_tensor = mytorch.Tensor(shape, device)
        mytorch.matrix_init(my_tensor)

        # Compare data
        my_tensor.cpu()
        my_tensor_data = np.array(my_tensor.data[:my_tensor.get_size()])
        torch_tensor_data = MyTensorToTorchTensor(my_tensor.copy()).numpy()

        np.testing.assert_allclose(my_tensor_data, torch_tensor_data, rtol=1e-5, atol=1e-8)

    def test_fc_forward(self):
        # Example shapes
        input_shape = [10, 45]
        weight_shape = [45, 4]
        bias_shape = [4]

        # Initialize custom Tensors
        input_tensor = mytorch.Tensor(input_shape, "GPU")
        weight_tensor = mytorch.Tensor(weight_shape, "GPU")
        bias_tensor = mytorch.Tensor(bias_shape, "GPU")
        output_tensor = mytorch.Tensor([10, 4], "GPU")
        mytorch.matrix_init(input_tensor)
        mytorch.matrix_init(weight_tensor)
        mytorch.matrix_init(bias_tensor)
        mytorch.matrix_init(output_tensor)

        # Initialize PyTorch tensors
        # tensor = input_tensor.copy()
        input_torch = MyTensorToTorchTensor(input_tensor.copy())
        weight_torch = MyTensorToTorchTensor(weight_tensor.copy())
        bias_torch = MyTensorToTorchTensor(bias_tensor.copy())

        # print(weight_tensor.shape)
        # print(weight_torch.shape)
        # Custom forward pass
        mytorch.fc_forward(input_tensor, weight_tensor, bias_tensor, output_tensor)

        # PyTorch forward pass
        torch_output = torch.nn.functional.linear(input_torch, weight_torch.T, bias_torch)

        # Compare data
        output_tensor.cpu()
        my_output_data = np.array(output_tensor.data[:output_tensor.get_size()])
        torch_output_data = torch_output.detach().numpy()

        np.testing.assert_allclose(my_output_data, torch_output_data, rtol=1e-5, atol=1e-8)

    def test_fc_backward(self):
        # Example shapes
        input_shape = [10, 45]
        weight_shape = [45, 4]
        bias_shape = [4]
        grad_output_shape = [10, 4]
        grad_input_shape = [10, 45]
        grad_weight_shape = [45, 4]

        # Initialize custom Tensors
        input_tensor = mytorch.Tensor(input_shape, "GPU")
        weight_tensor = mytorch.Tensor(weight_shape, "GPU")
        bias_tensor = mytorch.Tensor(bias_shape, "GPU")
        grad_output_tensor = mytorch.Tensor(grad_output_shape, "GPU")
        grad_input_tensor = mytorch.Tensor(grad_input_shape, "GPU")
        grad_weight_tensor = mytorch.Tensor(grad_weight_shape, "GPU")
        mytorch.matrix_init(input_tensor)
        mytorch.matrix_init(weight_tensor)
        mytorch.matrix_init(bias_tensor)
        mytorch.matrix_init(grad_output_tensor)
        mytorch.matrix_init(grad_input_tensor)
        mytorch.matrix_init(grad_weight_tensor)

        # Initialize PyTorch tensors
        input_torch = MyTensorToTorchTensor(input_tensor.copy()).requires_grad_(True)
        weight_torch = MyTensorToTorchTensor(weight_tensor.copy()).requires_grad_(True)
        bias_torch = MyTensorToTorchTensor(bias_tensor.copy()).requires_grad_(True)
        grad_output_torch = MyTensorToTorchTensor(grad_output_tensor.copy())

        # Custom backward pass
        mytorch.fc_backward(grad_output_tensor, input_tensor, weight_tensor, grad_weight_tensor, grad_input_tensor)

        # PyTorch backward pass
        torch_output = torch.nn.functional.linear(input_torch, weight_torch.T, bias_torch)
        torch_output.backward(grad_output_torch)

        # Compare data
        grad_input_tensor.cpu()
        grad_weight_tensor.cpu()
        my_grad_input_data = np.array(grad_input_tensor.data[:grad_input_tensor.get_size()])
        my_grad_weight_data = np.array(grad_weight_tensor.data[:grad_weight_tensor.get_size()])
        torch_grad_input_data = input_torch.grad.detach().numpy()
        torch_grad_weight_data = weight_torch.grad.detach().numpy()

        np.testing.assert_allclose(my_grad_input_data, torch_grad_input_data, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(my_grad_weight_data, torch_grad_weight_data, rtol=1e-5, atol=1e-8)



    def test_conv_forward(self):
        # Example shapes
        input_shape = [5, 3, 32, 32]
        weight_shape = [6, 3, 3, 3]
        output_shape = [5, 6, 32, 32]

        # Initialize custom Tensors
        input_tensor = mytorch.Tensor(input_shape, "GPU")
        weight_tensor = mytorch.Tensor(weight_shape, "GPU")
        output_tensor = mytorch.Tensor(output_shape, "GPU")
        mytorch.matrix_init(input_tensor)
        mytorch.matrix_init(weight_tensor)

        # Initialize PyTorch tensors
        input_torch = MyTensorToTorchTensor(input_tensor.copy())
        weight_torch = MyTensorToTorchTensor(weight_tensor.copy())

        # Custom forward pass
        mytorch.conv_forward(input_tensor, weight_tensor, output_tensor)

        # PyTorch forward pass
        torch_output = torch.nn.functional.conv2d(input_torch, weight_torch, padding=1)

        # Compare data
        output_tensor.cpu()
        my_output_data = np.array(output_tensor.data[:output_tensor.get_size()])
        torch_output_data = torch_output.detach().numpy()

        np.testing.assert_allclose(my_output_data, torch_output_data, rtol=1e-5, atol=1e-8)

    def test_conv_backward(self):
        # Example shapes
        input_shape = [1, 3, 32, 32]
        weight_shape = [6, 3, 3, 3]
        grad_output_shape = [1, 6, 32, 32]

        # Initialize custom Tensors
        input_tensor = mytorch.Tensor(input_shape, "GPU")
        weight_tensor = mytorch.Tensor(weight_shape, "GPU")
        grad_output_tensor = mytorch.Tensor(grad_output_shape, "GPU")
        grad_input_tensor = mytorch.Tensor(input_shape, "GPU")
        grad_weight_tensor = mytorch.Tensor(weight_shape, "GPU")
        mytorch.matrix_init(input_tensor)
        mytorch.matrix_init(weight_tensor)
        mytorch.matrix_init(grad_output_tensor)

        # Initialize PyTorch tensors
        input_torch = MyTensorToTorchTensor(input_tensor.copy()).requires_grad_(True)
        weight_torch = MyTensorToTorchTensor(weight_tensor.copy()).requires_grad_(True)
        grad_output_torch = MyTensorToTorchTensor(grad_output_tensor.copy())

        # Custom backward pass
        mytorch.conv_backward(grad_output_tensor, input_tensor, weight_tensor, grad_weight_tensor, grad_input_tensor)

        # PyTorch backward pass
        torch_output = torch.nn.functional.conv2d(input_torch, weight_torch, padding=1)
        torch_output.backward(grad_output_torch)

        # Compare data
        grad_input_tensor.cpu()
        grad_weight_tensor.cpu()
        my_grad_input_data = np.array(grad_input_tensor.data[:grad_input_tensor.get_size()])
        my_grad_weight_data = np.array(grad_weight_tensor.data[:grad_weight_tensor.get_size()])
        torch_grad_input_data = input_torch.grad.detach().numpy()
        torch_grad_weight_data = weight_torch.grad.detach().numpy()

        np.testing.assert_allclose(my_grad_input_data, torch_grad_input_data, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(my_grad_weight_data, torch_grad_weight_data, rtol=1e-5, atol=1e-8)

    def test_maxpool_forward(self):
        # Example shapes
        input_shape = [2, 3, 32, 32]
        output_shape = [2, 3, 16, 16]

        # Initialize custom Tensors
        input_tensor = mytorch.Tensor(input_shape, "GPU")
        output_tensor = mytorch.Tensor(output_shape, "GPU")
        mask_tensor = mytorch.Tensor(output_shape, "GPU")
        mytorch.matrix_init(input_tensor)

        # Initialize PyTorch tensors
        input_torch = MyTensorToTorchTensor(input_tensor.copy())

        # Custom forward pass
        mytorch.maxpool_forward(input_tensor, output_tensor, mask_tensor)

        # PyTorch forward pass
        torch_output = torch.nn.functional.max_pool2d(input_torch, kernel_size=2)

        # Compare data
        output_tensor.cpu()
        my_output_data = np.array(output_tensor.data[:output_tensor.get_size()])
        torch_output_data = torch_output.detach().numpy()

        np.testing.assert_allclose(my_output_data, torch_output_data, rtol=1e-5, atol=1e-8)

    def test_maxpool_backward(self):
        # Example shapes
        input_shape = [1, 3, 32, 32]
        grad_output_shape = [1, 3, 16, 16]
        output_shape = [1, 3, 16, 16]
        mask_shape = [1, 3, 16, 16]

        # Initialize custom Tensors
        input_tensor = mytorch.Tensor(input_shape, "GPU")
        mask_tensor = mytorch.Tensor(mask_shape, "GPU")
        output_tensor = mytorch.Tensor(output_shape, "GPU")
        grad_output_tensor = mytorch.Tensor(grad_output_shape, "GPU")
        grad_input_tensor = mytorch.Tensor(input_shape, "GPU")
        mytorch.matrix_init(input_tensor)
        mytorch.matrix_init(grad_output_tensor)

        mytorch.maxpool_forward(input_tensor, output_tensor, mask_tensor)
        mytorch.maxpool_backward(grad_output_tensor, mask_tensor, grad_input_tensor)

        # Initialize PyTorch tensors
        input_torch = MyTensorToTorchTensor(input_tensor.copy()).requires_grad_(True)
        grad_output_torch = MyTensorToTorchTensor(grad_output_tensor.copy())

        # PyTorch backward pass
        torch_output = torch.nn.functional.max_pool2d(input_torch, kernel_size=2)
        torch_output.backward(grad_output_torch)

        # Compare data
        grad_input_tensor.cpu()
        my_grad_input_data = np.array(grad_input_tensor.data[:grad_input_tensor.get_size()])
        torch_grad_input_data = input_torch.grad.detach().numpy()

        np.testing.assert_allclose(my_grad_input_data, torch_grad_input_data, rtol=1e-5, atol=1e-8)

    def test_softmax_forward(self):
        # Example shapes
        input_shape = [4, 50]
        output_shape = [4, 50]

        # Initialize custom Tensors
        input_tensor = mytorch.Tensor(input_shape, "GPU")
        output_tensor = mytorch.Tensor(output_shape, "GPU")
        mytorch.matrix_init(input_tensor)

        # Initialize PyTorch tensors
        input_torch = MyTensorToTorchTensor(input_tensor.copy())

        # Custom forward pass
        mytorch.softmax_forward(input_tensor, output_tensor)

        # PyTorch forward pass
        torch_output = torch.nn.functional.softmax(input_torch, dim=-1)

        # Compare data
        output_tensor.cpu()
        my_output_data = np.array(output_tensor.data[:output_tensor.get_size()])
        torch_output_data = torch_output.detach().numpy()

        np.testing.assert_allclose(my_output_data, torch_output_data, rtol=1e-5, atol=1e-8)

    def test_cross_entropy_forward(self):
        # Example shapes
        input_shape = [2, 10]
        target_shape = [2, 10]
        output_shape = [1]

        # Initialize custom Tensors
        input_tensor_unnormailized = mytorch.Tensor(input_shape, "GPU")
        input_tensor = mytorch.Tensor(input_shape, "GPU")
        target_tensor = mytorch.Tensor(target_shape, "CPU")
        output_tensor = mytorch.Tensor(output_shape, "GPU")
        mytorch.matrix_init(input_tensor_unnormailized)
        # mytorch.matrix_init(target_tensor)
        target_tensor.fill_(0)
        target_tensor.data[1] = 1
        target_tensor.gpu()
        target_tensor.print()

        mytorch.softmax_forward(input_tensor_unnormailized, input_tensor)
        input_tensor.print()

        # Initialize PyTorch tensors
        input_torch = MyTensorToTorchTensor(input_tensor.copy())
        target_torch =  MyTensorToTorchTensor(target_tensor.copy())
        # print(target_torch)

        # Custom forward pass
        mytorch.cross_entropy_forward(input_tensor, target_tensor, output_tensor)

        # PyTorch forward pass
        torch_output = torch.nn.functional.cross_entropy(input_torch, target_torch, reduction='sum')

        # Compare data
        output_tensor.cpu()
        my_output_data = np.array(output_tensor.data[:output_tensor.get_size()])
        torch_output_data = torch_output.detach().numpy()

        np.testing.assert_allclose(my_output_data, torch_output_data, rtol=1e-5, atol=1e-8)

    def test_cross_entropy_with_softmax_backward(self):
        # Example shapes
        input_shape = [2, 10]
        target_shape = [2]
        grad_output_shape = [2, 10]

        # Initialize custom Tensors
        input_tensor = mytorch.Tensor(input_shape, "GPU")
        target_tensor = mytorch.Tensor(target_shape, "GPU")
        grad_output_tensor = mytorch.Tensor(grad_output_shape, "GPU")
        mytorch.matrix_init(input_tensor)
        mytorch.matrix_init(target_tensor)

        # Initialize PyTorch tensors
        input_torch = MyTensorToTorchTensor(input_tensor.copy()).requires_grad_(True)
        target_torch = MyTensorToTorchTensor(target_tensor.copy())

        # Custom backward pass
        mytorch.cross_entropy_with_softmax_backward(input_tensor, target_tensor, grad_output_tensor)

        # PyTorch backward pass
        torch_output = torch.nn.functional.cross_entropy(input_torch, target_torch)
        torch_output.backward()

        # Compare data
        my_grad_output_data = np.array(grad_output_tensor.data)
        torch_grad_output_data = input_torch.grad.detach().numpy()

        np.testing.assert_allclose(my_grad_output_data, torch_grad_output_data, rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    unittest.main()