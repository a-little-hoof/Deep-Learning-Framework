from torchvision import datasets
from tqdm import tqdm
import os


train_data = datasets.MNIST(root="./data/", train=True, download=False)
test_data = datasets.MNIST(root="./data/", train=False, download=False)
saveDirTrain = './mnist/train'
saveDirTest = './mnist/test'

if not os.path.exists(saveDirTrain):
    os.mkdir(saveDirTrain)
if not os.path.exists(saveDirTest):
    os.mkdir(saveDirTest)



def save_img(data, save_path):
    for i in tqdm(range(len(data))):
        img, label = data[i]
        img.save(os.path.join(save_path, str(i) + '-label-' + str(label) + '.png'))


save_img(train_data, saveDirTrain)
save_img(test_data, saveDirTest)
