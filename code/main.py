import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from CelebADataset import CelebADataset
import VGG16
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Current Device: {device}\n', flush=True)

# 1. load the data
TRAIN_CSV = os.path.dirname(os.path.realpath(__file__)) + '/../data/celeba-train.csv'
VAL_CSV = os.path.dirname(os.path.realpath(__file__)) + '/../data/celeba-val.csv'
TEST_CSV = os.path.dirname(os.path.realpath(__file__)) + '/../data/celeba-test.csv'
ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../data/img_align_celeba'

# reference: https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg16-celeba.ipynb
# celebA image shape = (218, 178, 3)
# VGG-16 accepts (224, 224, 3) image by default
# adaptive pool helps with the variable-sized input
MEAN = (0.5063, 0.4258, 0.3832)
STD = (0.3107, 0.2904, 0.2897)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # not cropped:
    # (tensor([0.5063, 0.4258, 0.3832]), tensor([0.3107, 0.2904, 0.2897]))
    transforms.Normalize(MEAN, STD)
    # cropped: 
    # transforms.Normalize((0.5084, 0.4224, 0.3768), (0.3049, 0.2824, 0.2809))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
    #transforms.Normalize((0.5084, 0.4224, 0.3768), (0.3049, 0.2824, 0.2809))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
    #transforms.Normalize((0.5084, 0.4224, 0.3768), (0.3049, 0.2824, 0.2809))
])

# GOOGLE COLAB: CHANGE BATCH_SIZE
BATCH_SIZE = 128
NUM_WORKERS = 0

trainset = CelebADataset(TRAIN_CSV, ROOT_DIR, train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

valset = CelebADataset(VAL_CSV, ROOT_DIR, val_transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

testset = CelebADataset(TEST_CSV, ROOT_DIR, test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f'Total training data: {len(trainset)}', flush=True)
print(f'Total validation data: {len(valset)}', flush=True)
print(f'Total testing data: {len(testset)}', flush=True)
print(f'Total data: {len(trainset) + len(valset) + len(testset)}\n', flush=True)

# reference: https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/8
# import sys
# def online_mean_and_sd(loader):
#     """Compute the mean and sd in an online fashion

#         Var[x] = E[X^2] - E^2[X]
#     """
#     cnt = 0
#     fst_moment = torch.empty(3)
#     snd_moment = torch.empty(3)

#     for data in tqdm(loader):
#         data, _ = data
#         b, c, h, w = data.shape
#         nb_pixels = b * h * w
#         sum_ = torch.sum(data, dim=[0, 2, 3])
#         sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
#         fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
#         snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

#         cnt += nb_pixels

#     return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
# print(online_mean_and_sd(trainloader))
# sys.exit(1)
# (tensor([0.5084, 0.4224, 0.3768]), tensor([0.3049, 0.2824, 0.2809]))

# getting total images via the summation of length of each loader then * batch_size is invalid
# as total images might not be divisible by batch_size

# 2. instantiate the model
net = VGG16.Net(num_classes=2)

# reference: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html#create-model-and-dataparallel
if torch.cuda.device_count() > 1:
    print(f'Number of GPUs: {torch.cuda.device_count()}', flush=True)
    net = nn.DataParallel(net)

net.to(device)

# 3. define the loss function
criterion = nn.CrossEntropyLoss()

# 4. define the optimizer
# reference: https://medium.com/@youebned/notes-on-training-vgg16-7ae99689fd5
# following the paper to use momentum instead of adam optimiser
LEARNING_RATE = 0.001
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

#######################################################################
def compute_accuracy(net, dataloader):
    correct = 0
    total = 0
    for data in tqdm(dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total += len(labels)
    return correct / total * 100
#######################################################################

# 5. train the model
#######################################################################
# GOOGLE COLAB: CHANGE MODEL_DIRPATH
# path to directory where the checkpoint will be stored
MODEL_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../model/'
# for google colab
# MODEL_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../../drive/My Drive/VGG-16/model/'
CONTINUE_TRAIN = False
CONTINUE_TRAIN_NAME = MODEL_DIRPATH + 'model-epoch10.pth'
# since next_epoch store the next epoch value, we just need to deduct it from EPOCH without adding 1
EPOCH = 50
# save the model every SAVE_INTERVAL epoch
SAVE_INTERVAL = 5
########################################################################

next_epoch = 0
if CONTINUE_TRAIN:
    checkpoint = torch.load(CONTINUE_TRAIN_NAME)
    net.load_state_dict(checkpoint.get('net_state_dict'))
    optimizer.load_state_dict(checkpoint.get('optimizer_state_dict'))
    print(f"Last validation accuracy: {checkpoint.get('valacc')}%\n", flush=True)
    next_epoch = checkpoint.get('epoch')

for epoch in range(next_epoch, EPOCH):
    running_loss = 0.0
    net.train()
    print(f'Currently training: {net.training}', flush=True)
    for train_data in tqdm(trainloader, desc=f'Epoch {epoch + 1}/{EPOCH}'):
        inputs, labels = train_data[0].to(device), train_data[1].to(device)
        # 5a. zero the gradients
        optimizer.zero_grad()
        # 5b. forward propagation
        outputs = net(inputs)
        # 5c. compute loss
        loss = criterion(outputs, labels)
        # 5d. backward propagation
        loss.backward()
        # 5e. update parameters
        optimizer.step()

        running_loss += loss.item()

    # validation step
    net.eval()
    print(f'Currently training: {net.training}', flush=True)
    with torch.no_grad():
        # trainacc = compute_accuracy(net, trainloader)
        valacc = compute_accuracy(net, valloader)
    
    # save the trained model every 10 epochs
    # reference: https://discuss.pytorch.org/t/how-resume-the-saved-trained-model-at-specific-epoch/35823/3
    # reference: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    if (epoch + 1) % SAVE_INTERVAL == 0:
        torch.save({
            # since the currect epoch has been completed, save the next epoch
            'epoch' : epoch + 1,
            'net_state_dict' : net.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'valacc' : valacc
        }, MODEL_DIRPATH + f'model-epoch{epoch + 1}.pth')

    print(f'Training Loss: {running_loss / len(trainloader)}')
    # print(f'Training Accuracy: {trainacc}%')
    print(f'Validation Accuracy: {valacc}%')

# 6. save the trained model
torch.save({
            # since the currect epoch has been completed, save the next epoch
            'epoch' : EPOCH,
            'net_state_dict' : net.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'valacc' : valacc
        }, MODEL_DIRPATH + f'model-epoch{EPOCH}.pth')

# 7 . test the network
net.eval()
print(f'Currently training: {net.training}', flush=True)
testacc = compute_accuracy(net, testloader)

print(f'Testing Accuracy: {testacc}%', flush=True)