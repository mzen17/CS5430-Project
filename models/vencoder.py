# Variational Encoder Models
import torch
import torch.nn as nn
import torch.nn.functional as F

# q(z|x)
# given data, predict mean and var of q
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.h = F.relu
        self.fc2 = nn.Linear(256, 64)
        self.h2 = F.relu

        self.mean = nn.Linear(64, 8) # mean dist
        self.logvar = nn.Linear(64, 8) # <- it is supposed to output a log of the variance
        # this is to prevent the stuff from becoming <0

    def forward(self, input_tensor):
        out = self.h(self.fc1(input_tensor))
        out = self.h2(self.fc2(out))
        mean = self.mean(out)
        logvar = self.logvar(out) # <- it is supposed to output a log of the variance
        # this is to prevent the stuff from becoming <0

        return mean, logvar

# q(z|x)
# given data, predict mean and var of q
class ClassedEncoder(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        self.fc1 = nn.Linear(784 + class_count, 256)
        self.h = F.relu
        self.fc2 = nn.Linear(256, 64)
        self.h2 = F.relu

        self.mean = nn.Linear(64, 8) # mean dist
        self.logvar = nn.Linear(64, 8) # <- it is supposed to output a log of the variance
        # this is to prevent the stuff from becoming <0

    def forward(self, input_tensor, class_one_hot):
        out = self.h(self.fc1(torch.cat([input_tensor, class_one_hot], dim=1)))
        out = self.h2(self.fc2(out))
        mean = self.mean(out)
        logvar = self.logvar(out) # <- it is supposed to output a log of the variance
        # this is to prevent the stuff from becoming <0

        return mean, logvar
