import torch
import torch.nn as nn
import torch.nn.functional as F 

# VGG16 configuration
KERNEL_SIZE = (3, 3)
STRIDE = 1

class Net(nn.Module):

    def __init__(self, input_size=224, num_classes=1000):
        """
        input_size: int, size of input image, by default it is set to 224
        num_classes: int, total number of classes that an input image may belong to, by default it is set to 1000
        """
        super(Net, self).__init__()
        self.features = nn.Sequential(

        )
        self.classifier = nn.Sequential(
            # FC14
            # input size: input_size * input_size * 512
            # output size: 4096
            # Activation: ReLU
            nn.Linear(input_size * input_size * 512, 4096),
            nn.ReLU(),
            # FC15
            # input size: 4096
            # output size: 4096
            # Activation: ReLU
            nn.Linear(4096, 4096),
            nn.ReLU(),
            # FC16
            # input size: 4096
            # output size: num_classes
            # Activation: softmax
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        pass


if __name__ == "__main__":
    pass 