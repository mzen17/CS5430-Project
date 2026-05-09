## ELBO Loss, Minibatch
import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from models.vencoder import Encoder
from models.vdecoder import Decoder

# ----- DATA LOADING ----- #
mnist_data = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transforms.ToTensor()
)

all_images = mnist_data.data.float()
all_labels = mnist_data.targets
image_data = all_images.flatten(start_dim=1) / 255.0 # <- flatten the guy to linear layer.
label_data = F.one_hot(all_labels, num_classes=10).float()


print(image_data.size()) # 6742 784

# ----- TRAINING --------#
epochs = 10000 # <- actually mini batches but.
batch_size = 64
lr = 1e-3

encoder = Encoder(10, 8)
decoder = Decoder(8, 10)

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
    eps = torch.randn_like(z_std)
    zvals = (z_mean + eps * z_std)

    # p(x|z)
    x_mean, x_logvar = decoder.forward(zvals, label_batch) # generate our xdist from ptheta(x | z)
    x_std = torch.exp(0.5 * x_logvar) # same thing for q(z|x) here

    # ELBO, dim=8
    # we use K = 1
    # ELBO = p(x, z)/q(z|x)
    # log(ELBO) = p(z|x) + p(z) - q(z|x)
    FST = -8 * 0.5 * math.log(2 * 3.14159)
    FULLST = -784 * 0.5 * math.log(2 * 3.14159)

    # top head p(x, z) = p(x|z) + p(z)
    # p(z) is given by N(0, I). log(p(z)) = - k/2 ln[2pi] - 1/2 ln[det(cov)] - 1/2 sum([x]^2)
    # p(x|z) is given by N(x_m, x_v) => log(p(x|z)) = - k/2 ln [2pi] - 1/2 ln[det(cov)] - 1/2 sum((x-u)^2/b)
    pz = FST - 0.5 * torch.sum(zvals ** 2, dim=1)
    det_xvar = torch.sum(x_logvar, dim =1) # sum since we took log
    pxz = FULLST - 0.5 * det_xvar - 0.5 * torch.sum((batch - x_mean)**2/x_std**2, dim=1)

    # q(z|x) is given by pdf(z) from N(mean, var)
    det_zvar = torch.sum(z_logvar, dim=1) # sum since we took log
    qzx = FST - 0.5 * det_zvar - 0.5 * torch.sum((zvals - z_mean)**2/torch.exp(z_logvar), dim=1)

    negative_elbo  = torch.sum(-(pz + pxz - qzx))
    if ( i % 10 == 0): 
        print(f"Epoch {i} | {negative_elbo.item()}")

    negative_elbo.backward()
    encoder_optim.step()
    decoder_optim.step()

# sampling
# we generate 5 samples and decode the shit

generation = torch.tensor([1,2,5,3,4,4,1])
generation_onehot = F.one_hot(generation, num_classes=10).float()
SAMPLE_COUNT = len(generation)
z_vals = torch.randn(SAMPLE_COUNT, 8)


means, var = decoder.forward(z_vals, generation_onehot)

imgs = means.detach().cpu().view(SAMPLE_COUNT, 28, 28)

fig, axes = plt.subplots(1, SAMPLE_COUNT, figsize=(15, 3))

for i in range(SAMPLE_COUNT):
    axes[i].imshow(imgs[i], cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f"Sample {i+1}")

plt.tight_layout()
plt.savefig('img.png')
