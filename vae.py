## ELBO Loss, Minibatch
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from models.vencoder import Encoder
from models.vdecoder import Decoder
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

# ----- DATA LOADING ----- #
mnist_data = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transforms.ToTensor(),
)

all_images = mnist_data.data.float()
all_labels = mnist_data.targets
image_data = (all_images.flatten(start_dim=1) / 255.0).to(device) # <- flatten the guy to linear layer.
label_data = F.one_hot(all_labels, num_classes=10).float().to(device)

print(image_data.size()) # 60000 784

# ----- TRAINING --------#
epochs = 1000 # <- actually mini batches.
batch_size = 64
lr = 1e-3

encoder = Encoder(10, 8).to(device)
decoder = Decoder(8, 10, device).to(device)

encoder_optim = optim.Adam(encoder.parameters(), lr=lr)
decoder_optim = optim.Adam(decoder.parameters(), lr=lr)

for i in range(epochs):
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    indices = torch.randperm(image_data.size()[0])[:batch_size]
    batch = image_data[indices]
    label_batch = label_data[indices]

    # q(z|x)
    z_mean, z_logvar = encoder.forward(batch, label_batch) 

    # recall that the z output is log(variance)
    # to reverse it, we take the E
    # variance = e^(logvar)
    # to convert it to stdev, we have: sqrt(e^log(var)) = e^(0.5 logvar)
    z_std = torch.exp(0.5 * z_logvar)
    noise = torch.randn_like(z_std, device=device)
    zvals = (z_mean + noise * z_std)

    # p(x|z)
    x_mean, x_logvar = decoder.forward(zvals, label_batch) # generate our xdist from ptheta(x | z)
    x_std = torch.exp(0.5 * x_logvar) # same thing for q(z|x) here

    # ELBO, dim=8
    # ELBO = p(x|z) - dkl
    # dkl is - sum [(uj^2) + o^2 - 1 - log(oj^2)]
    pxz = 0.5 * (
        x_logvar + ((batch - x_mean) ** 2) / torch.exp(x_logvar)
    ).sum(dim=1).mean()

    dkl = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1).mean()
    negative_elbo = pxz + dkl

    if i % 10 == 0:
        print(f"epoch at {i} | DKL Loss: {negative_elbo}")

    negative_elbo.backward()
    encoder_optim.step()
    decoder_optim.step()

torch.save(encoder.state_dict(), output_dir / 'encoder.pt')
torch.save(decoder.state_dict(), output_dir / 'vae-decoder.pt')

# sampling
# we generate 5 samples and decode the shit

generation = torch.tensor([1,2,5,3,4,4,1], device=device)
generation_onehot = F.one_hot(generation, num_classes=10).float().to(device)
SAMPLE_COUNT = len(generation)
z_vals = torch.randn(SAMPLE_COUNT, 8, device=device)


means, var = decoder.forward(z_vals, generation_onehot)

imgs = means.detach().view(SAMPLE_COUNT, 28, 28)

fig, axes = plt.subplots(1, SAMPLE_COUNT, figsize=(15, 3))

for i in range(SAMPLE_COUNT):
    axes[i].imshow(imgs[i].cpu(), cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f"Sample {i+1}")

plt.tight_layout()
plt.savefig('vae-nums.png')
