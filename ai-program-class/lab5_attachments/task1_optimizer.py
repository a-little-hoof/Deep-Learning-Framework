"""
本文件我们尝试实现一个Optimizer类，用于优化一个简单的双层Linear Network
本次作业主要的内容将会在opti_epoch内对于一个epoch的参数进行优化
分为SGD_epoch和Adam_epoch两个函数，分别对应SGD和Adam两种优化器
其余函数为辅助函数，也请一并填写
和大作业的要求一致，我们不对数据处理和读取做任何要求
因此你可以引入任何的库来帮你进行数据处理和读取
理论上我们也不需要依赖lab5的内容，如果你需要的话，你可以将lab5对应代码copy到对应位置
"""
from task0_autodiff import *
from task0_operators import *
import numpy as np
from torchvision import datasets, transforms
import torch

t = 0
m = []
v = []

def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    """Generate all-zeros Tensor"""
    return constant(
        *shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )

def parse_mnist():
    """
    读取MNIST数据集，并进行简单的处理，如归一化
    你可以可以引入任何的库来帮你进行数据处理和读取
    所以不会规定你的输入的格式
    但需要使得输出包括X_tr, y_tr和X_te, y_te
    """
    ## 请于此填写你的代码
    data_path = "/data1/home/yifeiwang/Deep-Learning-Framework/ai-program-class/hw3/data"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.MNIST(root=data_path, train=True, download=False, transform=transform)
    test_data = datasets.MNIST(root=data_path, train=False, download=False, transform=transform)
    
    # Separate the data and labels
    X_tr = train_data.data.numpy().reshape(-1, 28*28) / 255.0
    y_tr = train_data.targets.numpy()
    X_te = test_data.data.numpy().reshape(-1, 28*28) / 255.0
    y_te = test_data.targets.numpy()
    return X_tr, y_tr, X_te, y_te

def set_structure(n, hidden_dim, k):
    """
    定义你的网络结构，并进行简单的初始化
    一个简单的网络结构为两个Linear层，中间加上ReLU
    Args:
        n: input dimension of the data.
        hidden_dim: hidden dimension of the network.
        k: output dimension of the network, which is the number of classes.
    Returns:
        List of Weights matrix.
    Example:
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)
    return list(W1, W2)
    """


    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)
    W1 = Tensor(W1)
    W2 = Tensor(W2)
    weights = [W1, W2]
    global m
    m = [Tensor(np.zeros(w.shape)) for w in weights]
    global v 
    v = [Tensor(np.zeros(w.shape)) for w in weights]
    return weights

    
    

def forward(X, weights):
    """
    使用你的网络结构，来计算给定输入X的输出
    Args:
        X : 2D input array of size (num_examples, input_dim).
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
    Returns:
        Logits calculated by your network structure.
    Example:
    W1 = weights[0]
    W2 = weights[1]
    return np.maximum(X@W1,0)@W2
    """
    
    W1 = weights[0]
    W2 = weights[1]

    XW1 = matmul(X, W1)
    XW1_relu = relu(XW1)
    output = matmul(XW1_relu, W2)
    return output

def softmax_loss(Z, y):
    """ 
    一个写了很多遍的Softmax loss...

    Args:
        Z : 2D numpy array of shape (batch_size, num_classes), 
        containing the logit predictions for each class.
        y : 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    # Compute the softmax
    Z = Z - broadcast_to(reshape(Tensor(np.max(Z.numpy(), axis=-1)), (Z.shape[0], 1)), Z.shape)
    # print(Z)
    Z_exp = exp(Z)
    # print(Z_exp)
    Z_sum = summation(Z_exp, axes=1)
    Z_sum = reshape(Z_sum, (Z_sum.shape[0], 1))
    softmax = Z_exp / broadcast_to(Z_sum, Z_exp.shape)
    # Compute the loss
    # label to one-hot
    y_onehot = np.zeros(Z.shape)
    y_onehot[np.arange(Z.shape[0]), y] = 1
    y_onehot = Tensor(y_onehot)
    # print(softmax)
    loss = negate(log(softmax+1e-30))
    loss = loss * y_onehot
    loss = summation(loss, axes=None)
    return loss / Z.shape[0]

def opti_epoch(X, y, weights, lr = 0.1, batch=100, beta1=0.9, beta2=0.999, using_adam=False):
    """
    优化一个epoch
    具体请参考SGD_epoch 和 Adam_epoch的代码
    """
    if using_adam:
        Adam_epoch(X, y, weights, lr = lr, batch=batch, beta1=beta1, beta2=beta2)
    else:
        SGD_epoch(X, y, weights, lr = lr, batch=batch)

def SGD_epoch(X, y, weights, lr = 0.1, batch=100):
    """ 
    SGD优化一个List of Weights
    本函数应该inplace地修改Weights矩阵来进行优化
    用学习率简单更新Weights

    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    # iterate over the epoch
    for i in range(0, X.shape[0], batch):
        X_batch = X[i:i+batch]
        y_batch = y[i:i+batch]
        # forward pass
        X_batch = Tensor(X_batch)
        # y_batch = Tensor(y_batch)
        Z = forward(X_batch, weights)
        
        # W1 = weights[0]
        # W2 = weights[1]

        # XW1 = matmul(X_batch, W1)
        # XW1_relu = relu(XW1)
        # Z = matmul(XW1_relu, W2)
        
        # backward pass
        loss = softmax_loss(Z, y_batch)
        # print(type(loss))
        loss.backward()
        # update weights
        # print(XW1_relu)
        # print(XW1.grad)
        # print(X_batch.grad)
        for w in weights:
            # print(w.grad)
            grad_np = w.grad.numpy()
            # print(np.max(grad_np), np.min(grad_np))
            w_np = w.numpy()
            # print(np.max(w_np), np.min(w_np))
            w.data = w.data - lr * w.grad
            w_new = w.numpy()
            # print(np.max(w_new-w_np), np.min(w_new-w_np))
        # exit(0)

def Adam_epoch(X, y, weights, lr = 0.1, batch=100, beta1=0.9, beta2=0.999):
    r"""
    ADAM优化一个
    本函数应该inplace地修改Weights矩阵来进行优化
    使用Adaptive Moment Estimation来进行更新Weights
    具体步骤可以是：
    1. 增加时间步 $t$。
    2. 计算当前梯度 $g$。
    3. 更新一阶矩向量：$m = \beta_1 \cdot m + (1 - \beta_1) \cdot g$。
    4. 更新二阶矩向量：$v = \beta_2 \cdot v + (1 - \beta_2) \cdot g^2$。
    5. 计算偏差校正后的一阶和二阶矩估计：$\hat{m} = m / (1 - \beta_1^t)$ 和 $\hat{v} = v / (1 - \beta_2^t)$。
    6. 更新参数：$\theta = \theta - \eta \cdot \hat{m} / (\sqrt{\hat{v}} + \epsilon)$。
    其中$\eta$表示学习率，$\beta_1$和$\beta_2$是平滑参数，
    $t$表示时间步，$\epsilon$是为了维持数值稳定性而添加的常数，如1e-8。
    
    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch
        beta1 (float): smoothing parameter for first order momentum
        beta2 (float): smoothing parameter for second order momentum

    Returns:
        None
    """
    global t
    global m
    global v
    epsilon = 1e-8
    # iterate over the epoch
    for i in range(0, X.shape[0], batch):
        X_batch = X[i:i+batch]
        y_batch = y[i:i+batch]

        # forward pass
        X_batch = Tensor(X_batch)
        # y_batch = Tensor(y_batch)
        Z = forward(X_batch, weights)
        
        # backward pass
        loss = softmax_loss(Z, y_batch)
        loss.backward()

        ## adam
        # update timestep
        t += 1
        # update m
        for i in range(len(m)):
            m[i].data = beta1 * m[i].data + (1 - beta1) * weights[i].grad
        # update v
        for i in range(len(v)):
            v[i].data = beta2 * v[i].data + (1 - beta2) * weights[i].grad ** 2

        m_hat = [m / (1 - beta1 ** t) for m in m]
        v_hat = [v / (1 - beta2 ** t) for v in v]
        # update weights
        for i,w in enumerate(weights):
            w.data -= lr * m_hat[i] / (Tensor(np.sqrt(v_hat[i].numpy())) + epsilon)

def loss_err(h,y):
    """ 
    计算给定预测结果h和真实标签y的loss和error
    """
    return softmax_loss(h,y), np.mean(h.numpy().argmax(axis=1) != y)


def train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100, beta1=0.9, beta2=0.999, using_adam=False):
    """ 
    训练过程
    """
    n, k = X_tr.shape[1], y_tr.max() + 1
    weights = set_structure(n, hidden_dim, k)
    np.random.seed(0)
    

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        opti_epoch(X_tr, y_tr, weights, lr=lr, batch=batch, beta1=beta1, beta2=beta2, using_adam=using_adam)
        train_loss, train_err = loss_err(forward(Tensor(X_tr), weights), y_tr)
        test_loss, test_err = loss_err(forward(Tensor(X_te), weights), y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss.numpy(), train_err, test_loss.numpy(), test_err))



if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = parse_mnist() 
    weights = set_structure(X_tr.shape[1], 100, y_tr.max() + 1)
    ## using SGD optimizer 
    train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim=100, epochs=20, lr = 0.001, batch=100, beta1=0.9, beta2=0.999, using_adam=False)
    ## using Adam optimizer
    # train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim=100, epochs=20, lr = 0.001, batch=100, beta1=0.9, beta2=0.999, using_adam=True)
    