import unittest
import torch
import numpy as np
import mytorch

import io
import sys
from contextlib import redirect_stdout

class TestTensor(unittest.TestCase):
    def test_tensor_creation(self):
        # Create a Tensor instance with the custom class
        shape = [2, 3]
        device = "CPU"
        my_tensor = mytorch.Tensor(shape, device)

        # Create an equivalent PyTorch tensor
        torch_tensor = torch.empty(shape, dtype=torch.float32)

        # Check shape and device
        self.assertEqual(my_tensor.shape, shape)
        self.assertEqual(my_tensor.device, device)
        self.assertEqual(my_tensor.get_size(), torch_tensor.numel())

    def test_fill_(self):
        # Test fill_ operation
        shape = [2, 3]
        value = 5.0

        my_tensor = mytorch.Tensor(shape, "CPU")
        my_tensor.fill_(value)

        torch_tensor = torch.full(shape, value, dtype=torch.float32)

        # Compare data
        my_tensor_data = np.array(my_tensor.data[:my_tensor.get_size()])
        torch_tensor_data = torch_tensor.numpy()

        np.testing.assert_allclose(my_tensor_data, torch_tensor_data)

    def test_gpu_conversion(self):
        # Test CPU to GPU and back
        shape = [2, 3]
        value = 3.14

        my_tensor = mytorch.Tensor(shape, "CPU")
        my_tensor.fill_(value)

        # Move to GPU
        my_tensor.gpu()
        self.assertEqual(my_tensor.device, "GPU")

        # Move back to CPU
        my_tensor.cpu()
        self.assertEqual(my_tensor.device, "CPU")

        # Verify data integrity after conversion
        my_tensor_data = np.array(my_tensor.data[:my_tensor.get_size()])
        expected_data = np.full(shape, value, dtype=np.float32)
        np.testing.assert_allclose(my_tensor_data, expected_data)

    def test_print(self):
        # Test the print functionality
        shape = [2, 3]
        value = 1.23

        my_tensor = mytorch.Tensor(shape, "CPU")
        my_tensor.fill_(value)

        # Redirect stdout to capture print output
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            my_tensor.print()

        # Verify printed values
        printed_output = captured_output.getvalue()
        expected_output = "1.230000 1.230000 1.230000 1.230000 1.230000 1.230000 \nshape: 2 3 \n"

if __name__ == "__main__":
    unittest.main()
