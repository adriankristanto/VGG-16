import torch
import torchvision
import torchvision.transforms as transforms 
import VGG16
from tqdm import tqdm
import os
import numpy as np
import PIL
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the test data
transform = transforms.Compose([
    transforms.CenterCrop((256, 256)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5084, 0.4224, 0.3768), (0.3049, 0.2824, 0.2809))
])

CLASSES = ['Female', 'Male']

def predict(image_path):
    image = PIL.Image.open(image_path)
    img_tensor = transform(image).unsqueeze_(0)
    # reference: https://github.com/pytorch/pytorch/issues/26338
    # functional dropout can't be turned off using net.eval()
    # they must be turned off manually
    net.eval()
    output = net(img_tensor)
    prob, index = torch.max(output, dim=1)
    return prob.item(), CLASSES[index.item()]

if __name__ == "__main__":
    # load the model
    FILEPATH = os.path.dirname(os.path.realpath(__file__)) + '/../model/'
    FILENAME = 'model-epoch10.pth'
    net = VGG16.Net(num_classes=2)
    # reference: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
    checkpoint = torch.load(FILEPATH + FILENAME, map_location=device)
    net.load_state_dict(checkpoint.get('net_state_dict'))
    # optimizer.load_state_dict(checkpoint.get('optimizer_state_dict'))
    # print(f"Last validation accuracy: {checkpoint.get('valacc')}%\n")
    # next_epoch = checkpoint.get('epoch')

    # predict image
    IMAGE_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../sample/'
    IMAGE_NAME = 'sample2.jpg'
    # image = PIL.Image.open(IMAGE_PATH + IMAGE_NAME)
    # image_tensor = transforms.ToTensor()(image)
    # print(image_tensor.shape)
    # plt.imshow(np.transpose(transforms.ToTensor()(PIL.Image.open(IMAGE_PATH + IMAGE_NAME)), (1,2,0)))
    # plt.show()
    prob, pred = predict(IMAGE_PATH + IMAGE_NAME)
    print(f'prediction: {prob * 100:.2f}% {pred}')