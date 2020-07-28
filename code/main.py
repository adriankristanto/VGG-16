import torch
import torchvision
import torchvision.transforms as transforms
import os
from CelebADataset import CelebADataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Current Device: {device}\n')

# 1. load the data
TRAIN_CSV = os.path.dirname(os.path.realpath(__file__)) + '/../data/celeba-train.csv'
VAL_CSV = os.path.dirname(os.path.realpath(__file__)) + '/../data/celeba-val.csv'
TEST_CSV = os.path.dirname(os.path.realpath(__file__)) + '/../data/celeba-test.csv'
ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../data/img_align_celeba'

# reference: https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg16-celeba.ipynb
# celebA image shape = (218, 178, 3)
# VGG-16 accepts (224, 224, 3) image by default
# we need to make the celebA images to be a square & divisible by 32
# divisible by 32 as the image will be passed through 5 MAX_POOL which will reduce the size by the factor of 2
train_transform = transforms.Compose([
    # take the smaller edge of the image (218, 178)
    transforms.CenterCrop((178, 178)),
    # 128/32 = 4
    # therefore, the output of the last POOL layer would be 4x4x512
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.CenterCrop((178, 178)),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.CenterCrop((178, 178)),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

BATCH_SIZE = 256
NUM_WORKERS = 0

trainset = CelebADataset(TRAIN_CSV, ROOT_DIR, train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

valset = CelebADataset(VAL_CSV, ROOT_DIR, val_transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

testset = CelebADataset(TEST_CSV, ROOT_DIR, test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)