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

    def __init__(self, num_classes=1000, init_params=True):
        """
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

        self.fc14 = nn.Linear(7*7*512, 4096)
        self.fc15 = nn.Linear(4096, 4096)
        self.fc16 = nn.Linear(4096, num_classes)

        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        # reference: https://discuss.pytorch.org/t/adaptive-avg-pool2d-vs-avg-pool2d/27011
        # (7,7) is the output height & width expected by the fully connected layers
        # in adaptive pooling, we can simply choose the output size & let pytorch determine
        # the kernel size, stride and padding
        self.adapt_pool = nn.AdaptiveAvgPool2d((7, 7))

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
        # reference: https://discuss.pytorch.org/t/adaptive-avg-pool2d-vs-avg-pool2d/27011
        # adaptive pooling can be used when the input size is variable
        # this is to ensure that the output of the last feature extractor
        # matches the input of the fully connected layer
        # no matter what the input image size is
        x = self.adapt_pool(x)
        # Flatten
        x = torch.flatten(x, start_dim=1)
        # FC14 -> ReLU -> FC15 -> ReLU -> FC16 -> Softmax
        x = self.fc14(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc15(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc16(x)
        # x = F.softmax(x, dim=1) 
        return x

    def initialise_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # reference: https://medium.com/ai%C2%B3-theory-practice-business/the-rectified-linear-unit-relu-and-kaiming-initialization-4c7a981dfd21
                # reference: https://stackoverflow.com/questions/61848635/how-to-decide-which-mode-to-use-for-kaiming-normal-initialization
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                # initialise the bias to 0
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)


if __name__ == "__main__":
    # test VGG16 configuration
    net = Net(num_classes=2)
    # net.initialise_weights()
    # import torchvision.models as models
    # vgg16 = models.vgg16()
    # num_features = vgg16.classifier[6].in_features
    # vgg16.classifier[6] = nn.Linear(num_features, 2)
    # from prettytable import PrettyTable
    # # reference: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model#:~:text=To%20get%20the%20parameter%20count,name%20and%20the%20parameter%20itself.
    # def count_parameters(model):
    #     table = PrettyTable(["Modules", "Parameters"])
    #     total_params = 0
    #     for name, parameter in model.named_parameters():
    #         if not parameter.requires_grad: continue
    #         param = parameter.numel()
    #         table.add_row([name, param])
    #         total_params+=param
    #     print(table)
    #     print(f"Total Trainable Params: {total_params}")
    #     return total_params
        
    # count_parameters(net)
    # count_parameters(vgg16)

    x = torch.rand([1,3,178,218])
    net.eval()
    print(net(x))
    # print(F.softmax(vgg16(x), dim=1))