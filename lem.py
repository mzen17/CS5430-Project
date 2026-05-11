## Langevin Approximation of Posterior, Minibatch
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from models.vencoder import Encoder
from models.vdecoder import Decoder

output_dir = Path('output')
encoder_path = output_dir / 'encoder.pt'
vae_decoder_path = output_dir / 'vae-decoder.pt'
z_bank_path = output_dir / 'z-bank.pt'

LOAD_FROM_ENCODER = True
LOAD_PREV_DECODER = True

# ----- DATA LOADING ----- #
mnist_data = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transforms.ToTensor()
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

all_images = mnist_data.data.float()
all_labels = mnist_data.targets
image_data = (all_images.flatten(start_dim=1) / 255.0).to(device) # <- flatten the guy to linear layer.
label_data = F.one_hot(all_labels, num_classes=10).float().to(device)

print(image_data.size())

# ----- TRAINING --------#
epochs = 1000
it_steps = 200
m_steps = 3
batch_size = 64
lr = 1e-3
lg_lr = 1e-2

encoder = Encoder(10, 8).to(device)
encoder.load_state_dict(torch.load(encoder_path, map_location=device))
encoder.eval()

decoder = Decoder(8, 10, device=device).to(device)
if LOAD_PREV_DECODER:
    decoder.load_state_dict(torch.load(vae_decoder_path, map_location=device))
decoder_optim = optim.Adam(decoder.parameters(), lr=lr)

with torch.no_grad():
    if LOAD_FROM_ENCODER:
        z_mean, z_logvar = encoder(image_data, label_data)
        z_std = torch.exp(0.5 * z_logvar)
        z_bank = z_mean + torch.randn_like(z_std) * z_std
    else:
        z_bank = torch.randn(len(image_data), 8, device=device)

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
    z_bank[indices] = z.detach()

    for m in range(m_steps):
        decoder_optim.zero_grad()

        model_output, logvar = decoder(z, label_batch)

        batch_loss = 0.5 * (
            logvar + ((batch - model_output) ** 2) / torch.exp(logvar)
        ).sum(dim=1).mean()
        batch_loss.backward()
        decoder_optim.step()

    if i % 10 == 0:
        print(f"Epoch {i} loss: {batch_loss}")

output_dir.mkdir(exist_ok=True)
torch.save(decoder.state_dict(), output_dir / 'lem-decoder.pt')
torch.save(
    {
        'z_bank': z_bank.detach().cpu(),
        'labels': all_labels.cpu(),
        'load_from_encoder': LOAD_FROM_ENCODER,
        'load_prev_decoder': LOAD_PREV_DECODER,
        'epochs': epochs,
        'it_steps': it_steps,
        'm_steps': m_steps,
        'lg_lr': lg_lr,
    },
    z_bank_path,
)

# sampling
generation = torch.tensor([1,2,5,3,4,4,1]).to(device)
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
with torch.no_grad():
    print("z_bank mean:", z_bank.mean(dim=0))
    print("z_bank std:", z_bank.std(dim=0))
    print("z_bank norm mean:", z_bank.norm(dim=1).mean())
    print("standard normal norm expected ~", 8 ** 0.5)
plt.tight_layout()
plt.savefig('img.png')
