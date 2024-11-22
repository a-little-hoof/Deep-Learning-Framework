import unittest
import torch
import numpy as np
import mytorch

class TestMyTorchFunctions(unittest.TestCase):

    def test_matrix_init(self):
        shape = [2, 3]
        device = "CPU"

        # Initialize custom Tensor
        my_tensor = mytorch.Tensor(shape, device)
        mytorch.matrix_init(my_tensor)

        # Initialize PyTorch tensor
        torch_tensor = torch.empty(shape, dtype=torch.float32)
        torch.nn.init.uniform_(torch_tensor, a=0.0, b=1.0)  # Example initialization

        # Compare data
        my_tensor_data = np.array(my_tensor.data)
        torch_tensor_data = torch_tensor.numpy()

        np.testing.assert_allclose(my_tensor_data, torch_tensor_data, rtol=1e-5, atol=1e-8)

    def test_fc_forward(self):
        # Example shapes
        input_shape = [2, 3]
        weight_shape = [3, 4]
        bias_shape = [4]

        # Initialize custom Tensors
        input_tensor = mytorch.Tensor(input_shape, "CPU")
        weight_tensor = mytorch.Tensor(weight_shape, "CPU")
        bias_tensor = mytorch.Tensor(bias_shape, "CPU")
        mytorch.matrix_init(input_tensor)
        mytorch.matrix_init(weight_tensor)
        mytorch.matrix_init(bias_tensor)

        # Initialize PyTorch tensors
        input_torch = torch.rand(input_shape, dtype=torch.float32)
        weight_torch = torch.rand(weight_shape, dtype=torch.float32)
        bias_torch = torch.rand(bias_shape, dtype=torch.float32)

        # Custom forward pass
        my_output = mytorch.fc_forward(input_tensor, weight_tensor, bias_tensor)

        # PyTorch forward pass
        torch_output = torch.nn.functional.linear(input_torch, weight_torch, bias_torch)

        # Compare data
        my_output_data = np.array(my_output.data)
        torch_output_data = torch_output.detach().numpy()

        np.testing.assert_allclose(my_output_data, torch_output_data, rtol=1e-5, atol=1e-8)

    def test_fc_backward(self):
        # Example shapes
        input_shape = [2, 3]
        weight_shape = [3, 4]
        bias_shape = [4]
        grad_output_shape = [2, 4]

        # Initialize custom Tensors
        input_tensor = mytorch.Tensor(input_shape, "CPU")
        weight_tensor = mytorch.Tensor(weight_shape, "CPU")
        bias_tensor = mytorch.Tensor(bias_shape, "CPU")
        grad_output_tensor = mytorch.Tensor(grad_output_shape, "CPU")
        mytorch.matrix_init(input_tensor)
        mytorch.matrix_init(weight_tensor)
        mytorch.matrix_init(bias_tensor)
        mytorch.matrix_init(grad_output_tensor)

        # Initialize PyTorch tensors
        input_torch = torch.rand(input_shape, dtype=torch.float32, requires_grad=True)
        weight_torch = torch.rand(weight_shape, dtype=torch.float32, requires_grad=True)
        bias_torch = torch.rand(bias_shape, dtype=torch.float32, requires_grad=True)
        grad_output_torch = torch.rand(grad_output_shape, dtype=torch.float32)

        # Custom backward pass
        my_grad_input, my_grad_weight, my_grad_bias = mytorch.fc_backward(input_tensor, weight_tensor, bias_tensor, grad_output_tensor)

        # PyTorch backward pass
        torch_output = torch.nn.functional.linear(input_torch, weight_torch, bias_torch)
        torch_output.backward(grad_output_torch)

        # Compare data
        my_grad_input_data = np.array(my_grad_input.data)
        my_grad_weight_data = np.array(my_grad_weight.data)
        my_grad_bias_data = np.array(my_grad_bias.data)
        torch_grad_input_data = input_torch.grad.detach().numpy()
        torch_grad_weight_data = weight_torch.grad.detach().numpy()
        torch_grad_bias_data = bias_torch.grad.detach().numpy()

        np.testing.assert_allclose(my_grad_input_data, torch_grad_input_data, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(my_grad_weight_data, torch_grad_weight_data, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(my_grad_bias_data, torch_grad_bias_data, rtol=1e-5, atol=1e-8)

    def test_conv_forward(self):
        # Example shapes
        input_shape = [1, 3, 32, 32]
        weight_shape = [6, 3, 5, 5]
        bias_shape = [6]
        output_shape = [1, 6, 28, 28]

        # Initialize custom Tensors
        input_tensor = mytorch.Tensor(input_shape, "CPU")
        weight_tensor = mytorch.Tensor(weight_shape, "CPU")
        bias_tensor = mytorch.Tensor(bias_shape, "CPU")
        output_tensor = mytorch.Tensor(output_shape, "CPU")
        mytorch.matrix_init(input_tensor)
        mytorch.matrix_init(weight_tensor)
        mytorch.matrix_init(bias_tensor)

        # Initialize PyTorch tensors
        input_torch = torch.rand(input_shape, dtype=torch.float32)
        weight_torch = torch.rand(weight_shape, dtype=torch.float32)
        bias_torch = torch.rand(bias_shape, dtype=torch.float32)

        # Custom forward pass
        mytorch.conv_forward(input_tensor, weight_tensor, bias_tensor, output_tensor)

        # PyTorch forward pass
        torch_output = torch.nn.functional.conv2d(input_torch, weight_torch, bias_torch)

        # Compare data
        my_output_data = np.array(output_tensor.data)
        torch_output_data = torch_output.detach().numpy()

        np.testing.assert_allclose(my_output_data, torch_output_data, rtol=1e-5, atol=1e-8)

    def test_conv_backward(self):
        # Example shapes
        input_shape = [1, 3, 32, 32]
        weight_shape = [6, 3, 5, 5]
        bias_shape = [6]
        grad_output_shape = [1, 6, 28, 28]

        # Initialize custom Tensors
        input_tensor = mytorch.Tensor(input_shape, "CPU")
        weight_tensor = mytorch.Tensor(weight_shape, "CPU")
        bias_tensor = mytorch.Tensor(bias_shape, "CPU")
        grad_output_tensor = mytorch.Tensor(grad_output_shape, "CPU")
        grad_input_tensor = mytorch.Tensor(input_shape, "CPU")
        grad_weight_tensor = mytorch.Tensor(weight_shape, "CPU")
        grad_bias_tensor = mytorch.Tensor(bias_shape, "CPU")
        mytorch.matrix_init(input_tensor)
        mytorch.matrix_init(weight_tensor)
        mytorch.matrix_init(bias_tensor)
        mytorch.matrix_init(grad_output_tensor)

        # Initialize PyTorch tensors
        input_torch = torch.rand(input_shape, dtype=torch.float32, requires_grad=True)
        weight_torch = torch.rand(weight_shape, dtype=torch.float32, requires_grad=True)
        bias_torch = torch.rand(bias_shape, dtype=torch.float32, requires_grad=True)
        grad_output_torch = torch.rand(grad_output_shape, dtype=torch.float32)

        # Custom backward pass
        mytorch.conv_backward(input_tensor, weight_tensor, bias_tensor, grad_output_tensor, grad_input_tensor, grad_weight_tensor, grad_bias_tensor)

        # PyTorch backward pass
        torch_output = torch.nn.functional.conv2d(input_torch, weight_torch, bias_torch)
        torch_output.backward(grad_output_torch)

        # Compare data
        my_grad_input_data = np.array(grad_input_tensor.data)
        my_grad_weight_data = np.array(grad_weight_tensor.data)
        my_grad_bias_data = np.array(grad_bias_tensor.data)
        torch_grad_input_data = input_torch.grad.detach().numpy()
        torch_grad_weight_data = weight_torch.grad.detach().numpy()
        torch_grad_bias_data = bias_torch.grad.detach().numpy()

        np.testing.assert_allclose(my_grad_input_data, torch_grad_input_data, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(my_grad_weight_data, torch_grad_weight_data, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(my_grad_bias_data, torch_grad_bias_data, rtol=1e-5, atol=1e-8)

    def test_maxpool_forward(self):
        # Example shapes
        input_shape = [1, 3, 32, 32]
        output_shape = [1, 3, 16, 16]

        # Initialize custom Tensors
        input_tensor = mytorch.Tensor(input_shape, "CPU")
        output_tensor = mytorch.Tensor(output_shape, "CPU")
        mytorch.matrix_init(input_tensor)

        # Initialize PyTorch tensors
        input_torch = torch.rand(input_shape, dtype=torch.float32)

        # Custom forward pass
        mytorch.maxpool_forward(input_tensor, output_tensor)

        # PyTorch forward pass
        torch_output = torch.nn.functional.max_pool2d(input_torch, kernel_size=2)

        # Compare data
        my_output_data = np.array(output_tensor.data)
        torch_output_data = torch_output.detach().numpy()

        np.testing.assert_allclose(my_output_data, torch_output_data, rtol=1e-5, atol=1e-8)

    def test_maxpool_backward(self):
        # Example shapes
        input_shape = [1, 3, 32, 32]
        grad_output_shape = [1, 3, 16, 16]

        # Initialize custom Tensors
        input_tensor = mytorch.Tensor(input_shape, "CPU")
        grad_output_tensor = mytorch.Tensor(grad_output_shape, "CPU")
        grad_input_tensor = mytorch.Tensor(input_shape, "CPU")
        mytorch.matrix_init(input_tensor)
        mytorch.matrix_init(grad_output_tensor)

        # Initialize PyTorch tensors
        input_torch = torch.rand(input_shape, dtype=torch.float32, requires_grad=True)
        grad_output_torch = torch.rand(grad_output_shape, dtype=torch.float32)

        # Custom backward pass
        mytorch.maxpool_backward(input_tensor, grad_output_tensor, grad_input_tensor)

        # PyTorch backward pass
        torch_output = torch.nn.functional.max_pool2d(input_torch, kernel_size=2)
        torch_output.backward(grad_output_torch)

        # Compare data
        my_grad_input_data = np.array(grad_input_tensor.data)
        torch_grad_input_data = input_torch.grad.detach().numpy()

        np.testing.assert_allclose(my_grad_input_data, torch_grad_input_data, rtol=1e-5, atol=1e-8)

    def test_softmax_forward(self):
        # Example shapes
        input_shape = [2, 10]
        output_shape = [2, 10]

        # Initialize custom Tensors
        input_tensor = mytorch.Tensor(input_shape, "CPU")
        output_tensor = mytorch.Tensor(output_shape, "CPU")
        mytorch.matrix_init(input_tensor)

        # Initialize PyTorch tensors
        input_torch = torch.rand(input_shape, dtype=torch.float32)

        # Custom forward pass
        mytorch.softmax_forward(input_tensor, output_tensor)

        # PyTorch forward pass
        torch_output = torch.nn.functional.softmax(input_torch, dim=1)

        # Compare data
        my_output_data = np.array(output_tensor.data)
        torch_output_data = torch_output.detach().numpy()

        np.testing.assert_allclose(my_output_data, torch_output_data, rtol=1e-5, atol=1e-8)

    def test_cross_entropy_forward(self):
        # Example shapes
        input_shape = [2, 10]
        target_shape = [2]
        output_shape = [1]

        # Initialize custom Tensors
        input_tensor = mytorch.Tensor(input_shape, "CPU")
        target_tensor = mytorch.Tensor(target_shape, "CPU")
        output_tensor = mytorch.Tensor(output_shape, "CPU")
        mytorch.matrix_init(input_tensor)
        mytorch.matrix_init(target_tensor)

        # Initialize PyTorch tensors
        input_torch = torch.rand(input_shape, dtype=torch.float32)
        target_torch = torch.randint(0, 10, target_shape, dtype=torch.long)

        # Custom forward pass
        mytorch.cross_entropy_forward(input_tensor, target_tensor, output_tensor)

        # PyTorch forward pass
        torch_output = torch.nn.functional.cross_entropy(input_torch, target_torch)

        # Compare data
        my_output_data = np.array(output_tensor.data)
        torch_output_data = torch_output.detach().numpy()

        np.testing.assert_allclose(my_output_data, torch_output_data, rtol=1e-5, atol=1e-8)

    def test_cross_entropy_with_softmax_backward(self):
        # Example shapes
        input_shape = [2, 10]
        target_shape = [2]
        grad_output_shape = [2, 10]

        # Initialize custom Tensors
        input_tensor = mytorch.Tensor(input_shape, "CPU")
        target_tensor = mytorch.Tensor(target_shape, "CPU")
        grad_output_tensor = mytorch.Tensor(grad_output_shape, "CPU")
        mytorch.matrix_init(input_tensor)
        mytorch.matrix_init(target_tensor)

        # Initialize PyTorch tensors
        input_torch = torch.rand(input_shape, dtype=torch.float32, requires_grad=True)
        target_torch = torch.randint(0, 10, target_shape, dtype=torch.long)

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