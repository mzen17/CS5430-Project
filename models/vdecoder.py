# Variational Auto Decoder Models
import torch
import torch.nn as nn
import torch.nn.functional as F

# p(x|z). Fixed variance output.
# works interestingly.
# given z, predict mean and var of p
class FixedVarDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(8, 64)
        self.h = F.relu

        self.mean = nn.Linear(64, 784)
        self.logvar = torch.zeros(1, 784)
        # NO VARIANCE
        # fixed variance of 1

        self.sigmoid = F.sigmoid
    
    def forward(self, latent):
        out = self.h(self.fc1(latent))
        mean = self.sigmoid(self.mean(out))
        # small transform since it doesn't make sense for weights to out 100

        return mean, self.logvar

class ClassedFixedVarDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(8, 64)
        self.h = F.relu

        self.mean = nn.Linear(64, 784)
        self.logvar = torch.zeros(1, 784)
        # NO VARIANCE
        # fixed variance of 1

        self.sigmoid = F.sigmoid
    
    def forward(self, latent):
        out = self.h(self.fc1(latent))
        mean = self.sigmoid(self.mean(out))
        # small transform since it doesn't make sense for weights to out 100

        return mean, self.logvar
