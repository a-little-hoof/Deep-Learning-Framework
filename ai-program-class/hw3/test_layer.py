import mytorch
import torch
import torch.nn as nn

import unittest


linear = nn.Linear(10, 5) # 输入特征维度为10，输出特征维度为5
x = torch.randn(2, 10)  # 2个样本，每个样本10维
output = linear(x)


