# Variational Auto Decoder Models
import torch
import torch.nn as nn
import torch.nn.functional as F

# p(x|z). Fixed variance output.
# works interestingly.
# given z, predict mean and var of p
class Decoder(nn.Module):
    def __init__(self, latent_dim, class_count, device="cpu"):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim + class_count, 64)
        self.h = F.relu
        self.fc2 = nn.Linear(64, 256)
        self.h2 = F.relu
        self.mean = nn.Linear(256, 784)
        self.logvar = torch.zeros(1, 784, device=device)
        # NO VARIANCE
        # fixed variance of 1

        self.sigmoid = F.sigmoid
    
    def forward(self, latent, class_type):
        out = self.h(self.fc1(torch.cat([latent, class_type], dim=1)))
        out = self.h2(self.fc2(out))
        mean = self.sigmoid(self.mean(out))
        # small transform since it doesn't make sense for weights to out 100

        return mean, self.logvar
