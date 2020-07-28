import torch
import torch.nn as nn
import torch.nn.functional as F 

# VGG16 configuration
CONV_KERNEL_SIZE = (3, 3)
CONV_STRIDE = (1, 1)
POOL_KERNEL_SIZE = (2, 2)
POOL_STRIDE = (2, 2)

def p(n, f, s, out):
    """
    n: input size
    f: kernel/filter size
    s: stride
    out: output size
    """
    # out = (n + 2p -f)/s + 1
    return (s * (out - 1) - n + f)/2

class Net(nn.Module):

    def __init__(self, input_size=224, num_classes=1000):
        """
        input_size: int, size of input image, by default it is set to 224
        num_classes: int, total number of classes that an input image may belong to, by default it is set to 1000
        """
        super(Net, self).__init__()
        self.features = nn.Sequential(
            # CONV1
            # input size: input_size * input_size * 3
            # output size: input_size * input_size * 64
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=p(input_size, CONV_KERNEL_SIZE, CONV_STRIDE, input_size)),
            nn.ReLU(),
            # CONV2
            # input size: input_size * input_size * 64
            # output size: input_size * input_size * 64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=p(input_size, CONV_KERNEL_SIZE, CONV_STRIDE, input_size)),
            nn.ReLU(),
            # POOL2
            # input size: input_size * input_size * 64
            # output size: input_size/2 * input_size/2 * 64
            nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE),
            # CONV3
            # input size: input_size/2 * input_size/2 * 64
            # output size: input_size/2 * input_size/2 * 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=p(input_size, CONV_KERNEL_SIZE, CONV_STRIDE, input_size)),
            nn.ReLU(),
            # CONV4
            # input size: input_size/2 * input_size/2 * 128
            # output size: input_size/2 * input_size/2 * 128
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=p(input_size, CONV_KERNEL_SIZE, CONV_STRIDE, input_size)),
            nn.ReLU(),
            # POOL4
            # input size: input_size/2 * input_size/2 * 128
            # output size: input_size/4 * input_size/4 * 128
            nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE),
            # CONV5
            # input size: input_size/4 * input_size/4 * 128
            # output size: input_size/4 * input_size/4 * 256
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=p(input_size, CONV_KERNEL_SIZE, CONV_STRIDE, input_size)),
            nn.ReLU(),
            # CONV6
            # input size: input_size/4 * input_size/4 * 256
            # output size: input_size/4 * input_size/4 * 256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=p(input_size, CONV_KERNEL_SIZE, CONV_STRIDE, input_size)),
            nn.ReLU(),
            # CONV7
            # input size: input_size/4 * input_size/4 * 256
            # output size: input_size/4 * input_size/4 * 256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=p(input_size, CONV_KERNEL_SIZE, CONV_STRIDE, input_size)),
            nn.ReLU(),
            # POOL7
            # input size: input_size/4 * input_size/4 * 256
            # output size: input_size/8 * input_size/8 * 256
            nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE),
            # CONV8
            # input size: input_size/8 * input_size/8 * 256
            # output size: input_size/8 * input_size/8 * 512
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=p(input_size, CONV_KERNEL_SIZE, CONV_STRIDE, input_size)),
            nn.ReLU(),
            # CONV9
            # input size: input_size/8 * input_size/8 * 512
            # output size: input_size/8 * input_size/8 * 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=p(input_size, CONV_KERNEL_SIZE, CONV_STRIDE, input_size)),
            nn.ReLU(),
            # CONV10
            # input size: input_size/8 * input_size/8 * 512
            # output size: input_size/8 * input_size/8 * 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=p(input_size, CONV_KERNEL_SIZE, CONV_STRIDE, input_size)),
            nn.ReLU(),
            # POOL10
            # input size: input_size/8 * input_size/8 * 512
            # output size: input_size/16 * input_size/16 * 512
            nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE),
            # CONV11
            # input size: input_size/16 * input_size/16 * 512
            # output size: input_size/16 * input_size/16 * 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=p(input_size, CONV_KERNEL_SIZE, CONV_STRIDE, input_size)),
            nn.ReLU(),
            # CONV12
            # input size: input_size/16 * input_size/16 * 512
            # output size: input_size/16 * input_size/16 * 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=p(input_size, CONV_KERNEL_SIZE, CONV_STRIDE, input_size)),
            nn.ReLU(),
            # CONV13
            # input size: input_size/16 * input_size/16 * 512
            # output size: input_size/16 * input_size/16 * 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=p(input_size, CONV_KERNEL_SIZE, CONV_STRIDE, input_size)),
            nn.ReLU(),
            # POOL13
            # input size: input_size/16 * input_size/16 * 512
            # output size: input_size/32 * input_size/32 * 512
            nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE)
        )
        self.classifier = nn.Sequential(
            # FC14
            # input size: input_size/32 * input_size/32 * 512
            # output size: 4096
            # Activation: ReLU
            nn.Linear(input_size/32 * input_size/32 * 512, 4096),
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
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    pass 