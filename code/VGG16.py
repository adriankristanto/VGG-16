import torch
import torch.nn as nn
import torch.nn.functional as F 

class Net(nn.module):

    def __init__(self):
        super(Net, self).__init__()