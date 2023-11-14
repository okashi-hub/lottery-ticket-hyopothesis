import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .model_module import PruningModule, MaskedLinear, MaskedConv2d
import torch.nn.init as init


class AntNet(PruningModule):
    def __init__(self, mask=False):
        super(AntNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        
        self.fc1 = linear(28*28, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)        
        
        self.fc4 = linear(28*28, 300)
        self.fc5 = linear(300, 100)
        self.fc6 = linear(100, 10)        
        
        self.fc7 = linear(28*28, 300)
        self.fc8 = linear(300, 100)
        self.fc9 = linear(100, 10)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear, MaskedLinear)):
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = x.view(-1, 28*28)
        
        x1 = x.copy.deepcopy()
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        x2 = x.copy.deepcopy()
        x2 = F.relu(self.fc4(x2))
        x2 = F.relu(self.fc5(x2))
        x2 = self.fc6(x2)

        x3 = x.copy.deepcopy()
        x3 = F.relu(self.fc7(x3))
        x3 = F.relu(self.fc8(x3))
        x3 = self.fc9(x3)
        
        return x1 + x2 + x3