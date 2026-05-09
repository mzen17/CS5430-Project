## Langevin Approximation of Posterior, Minibatch
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

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


print(image_data.size())

# ----- TRAINING --------#
epochs = 1000
it_steps = 100
batch_size = 64
lr = 1e-3
lg_lr = 1e-4

# these are commented out because we no longer use the encoder
# we only need the decoder
# encoder = Encoder(10, 8)
#encoder_optim = optim.Adam(encoder.parameters(), lr=lr)
decoder = Decoder(8, 10)
decoder_optim = optim.Adam(decoder.parameters(), lr=lr)

z_bank = torch.randn(len(image_data), 8) # <- persistent z banks for the guys

for i in range(epochs):
    decoder_optim.zero_grad()

    indices = torch.randperm(image_data.size()[0])[:batch_size]
    batch = image_data[indices]
    label_batch = label_data[indices]

    z = z_bank[indices].clone().detach().requires_grad_(True)
    
    #---- E STEP ----#
    for j in range(it_steps):
        z = z.detach().requires_grad_(True) # reset gradient calculations
        # this way we are capuring it only for this run.

        pxz_mean, pxz_logvar = decoder(z, label_batch) # p(x|z)

        # The langevin sampling is defined as:
        # z_t+1 = lr [nabla log p(z_t) + nabla z log p theta (x|z) + 2 sqrte]
        
        random_noise = torch.randn_like(z)

        pzt = -z # gradient of normal is negative z
        
        log_pxz = -0.5 * torch.sum((
            pxz_logvar + (pxz_mean - batch)**2/torch.exp(pxz_logvar)
            ), dim=1)
        grad_z = torch.autograd.grad(log_pxz.sum(), z)[0] # pull gradient out of decoder

        # iterative z step
        z = z + lg_lr * (pzt + grad_z) + random_noise * ((2 * lg_lr)**0.5)
    
    #---- M STEP ----#
    # decoder updates
    # with no encoder, we just do a Gaussian NLL instead of KLD
    z = z.detach()
    decoder_optim.zero_grad() # <- reset the guy just for safety

    model_output, logvar = decoder(z, label_batch)

    batch_loss = 0.5 * (
        logvar + ((batch - model_output) ** 2) / torch.exp(logvar)
    ).sum(dim=1).mean()
    batch_loss.backward()

    z_bank[indices] = z.detach()

    if i % 10 == 0:
        print(f"Epoch {i} loss: {batch_loss}")

    decoder_optim.step()

# sampling
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
