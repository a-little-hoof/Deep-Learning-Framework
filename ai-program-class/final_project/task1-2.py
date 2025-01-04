import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import os

### hyperparameters
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
epochs = 10
batch_size = 8192
lr = 5e-2
momentum = 0.9
device = "cuda:0"

### define network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Change the input channels to 1 (for grayscale images)
        self.conv1 = nn.Conv2d(1, 6, 5)  
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)  # Adjust input size for the fully connected layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # Output for 10 classes (digits 0-9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def run():

    writer = SummaryWriter("./tensorboard")

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])  # MNIST uses 1 channel, so mean and std are adjusted accordingly.

    ### load dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    net = Net()

    # ## task 2 data parallel
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     net = nn.DataParallel(net)

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    cnt = 0
    ## cal training time
    import time
    start = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # print(f"input shape: {inputs.shape}, output shape: {outputs.shape}")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss", loss.item(), cnt)
            cnt += 1
            running_loss += loss.item()
        print(f'[epoch: {epoch + 1}] loss: {running_loss / (len(trainset)/batch_size)}')
    end = time.time()
    print(f'Finished Training, use {end - start} seconds')

    ### test
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    classes = tuple(str(i) for i in range(10))  # MNIST classes are digits 0-9
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

if __name__ == "__main__":
    run()