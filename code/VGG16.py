import torch
import torch.nn as nn
import torch.nn.functional as F 

# reference: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
# padding calculation for CONV layer as POOL layer doesn't require padding
# each CONV has the same configuration
# s = 1, f = 3 and we want 'same' padding
# therefore,
"""
p = (s(out-1)-n+f) / 2
p = (out-1-n+f) / 2
same padding means n == out
p = (n-1-n+f) / 2
p = (f-1)/2
p = (3-1)/2 = 1
"""

class Net(nn.Module):

    def __init__(self, input_size=224, num_classes=1000, init_params=True):
        """
        input_size: int, size of input image, by default it is set to 224
        num_classes: int, total number of classes that an input image may belong to, by default it is set to 1000
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1)

        conv_output = (input_size // 32) * (input_size // 32) * 512
        self.fc14 = nn.Linear(conv_output, 4096)
        self.fc15 = nn.Linear(4096, 4096)
        self.fc16 = nn.Linear(4096, num_classes)

        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        if init_params:
            self.initialise_weights()

    def forward(self, x):
        # CONV1 -> ReLU -> CONV2 -> ReLU -> POOL
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        # CONV3 -> ReLU -> CONV4 -> ReLU -> POOL
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)
        # CONV5 -> ReLU -> CONV6 -> ReLU -> CONV7 -> ReLU -> POOL
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.pool(x)
        # CONV8 -> ReLU -> CONV9 -> ReLU -> CONV10 -> ReLU -> POOL
        x = self.conv8(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = F.relu(x)
        x = self.conv10(x)
        x = F.relu(x)
        x = self.pool(x)
        # CONV11 -> ReLU -> CONV12 -> ReLU -> CONV13 -> ReLU -> POOL
        x = self.conv11(x)
        x = F.relu(x)
        x = self.conv12(x)
        x = F.relu(x)
        x = self.conv13(x)
        x = F.relu(x)
        x = self.pool(x)
        # Flatten
        x = torch.flatten(x, start_dim=1)
        # FC14 -> ReLU -> FC15 -> ReLU -> FC16 -> Softmax
        x = self.fc14(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc15(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc16(x)
        x = F.softmax(x, dim=1)
        return x

    def initialise_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # initialise CONV layers' weights with xavier_uniform_
                nn.init.xavier_uniform_(module.weight)
                # initialise the bias to 0
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.contanst_(module.bias, 0)


if __name__ == "__main__":
    # test VGG16 configuration
    net = Net(input_size=128, num_classes=2)
    net.initialise_weights()
    # import torchvision.models as models
    # vgg16 = models.vgg16()
    # num_features = vgg16.classifier[6].in_features
    # vgg16.classifier[6] = nn.Linear(num_features, 2)
    # print(vgg16)
    # print(net)

    # x = torch.rand([1,3,128,128])
    # print(net(x))
    # print(F.softmax(vgg16(x), dim=1))