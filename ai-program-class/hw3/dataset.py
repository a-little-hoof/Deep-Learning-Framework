from torchvision import datasets, transforms
import torch
import numpy as np
import os
import mytorch
import unittest

class TestTensor(unittest.TestCase):
    def test_load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_data = datasets.MNIST(root="./data/", train=True, download=False, transform=transform)
        test_data = datasets.MNIST(root="./data/", train=False, download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
        data, labels = next(iter(trainloader))
        data = data.numpy()
        labels = labels.numpy()
        # 将 numpy 数组转换为自定义 Tensor
        images_tensor = mytorch.numpy_to_tensor(data)
        labels_tensor = mytorch.numpy_to_tensor(labels)

        self.assertEqual(images_tensor.shape, list(data.shape))
        self.assertEqual(labels_tensor.shape, list(labels.shape))

        # compare data
        images_tensor.cpu()
        images_tensor_data = np.array(images_tensor.data[:images_tensor.get_size()])
        labels_tensor.cpu()
        labels_tensor_data = np.array(labels_tensor.data[:labels_tensor.get_size()])
        np.testing.assert_allclose(images_tensor_data, data)
        np.testing.assert_allclose(labels_tensor_data, labels)

if __name__ == '__main__':
    unittest.main()