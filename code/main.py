import torch
import torchvision
import torchvision.transforms as transforms
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Current Device: {device}\n')

# 1. load the data
DATA_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../data'
BATCH_SIZE = 256
NUM_WORKERS = 0

# reference: https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg16-celeba.ipynb
train_transform = transforms.Compose([
    transform.ToTensor()
])

test_transform = transforms.Compose([
    transform.ToTensor()
])