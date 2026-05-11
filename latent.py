from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torchvision import datasets, transforms

from models.vdecoder import Decoder
from models.vencoder import Encoder


device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = Path('output')

encoder_path = output_dir / 'encoder.pt'
lem_decoder_path = output_dir / 'lem-decoder.pt'
z_bank_path = output_dir / 'z-bank.pt'
plot_path = 'latent-representations.png'
latents_path = output_dir / 'latent-representations.pt'

sample_count = 1000
it_samples = 50
lg_lr = 1e-4


def require_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the training scripts that creates the checkpoints first."
        )


def load_data():
    mnist_data = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    images = mnist_data.data[:sample_count].float().flatten(start_dim=1) / 255.0
    labels = mnist_data.targets[:sample_count]
    onehot_labels = F.one_hot(labels, num_classes=10).float()

    return images.to(device), labels, onehot_labels.to(device)


def encoder_latents(encoder, images, onehot_labels):
    encoder.eval()

    with torch.no_grad():
        z_mean, _ = encoder(images, onehot_labels)

    return z_mean


def langevin_latents(decoder, images, onehot_labels):
    decoder.eval()
    z = torch.randn(images.size(0), 8, device=device)

    for _ in range(it_samples):
        z = z.detach().requires_grad_(True)
        pxz_mean, pxz_logvar = decoder(z, onehot_labels)
        random_noise = torch.randn_like(z)

        prior_grad = -z
        log_pxz = -0.5 * torch.sum(
            pxz_logvar + (pxz_mean - images) ** 2 / torch.exp(pxz_logvar),
            dim=1,
        )
        likelihood_grad = torch.autograd.grad(log_pxz.sum(), z)[0]
        z = z + lg_lr * (prior_grad + likelihood_grad) + random_noise * (2 * lg_lr) ** 0.5

    return z.detach()


def load_z_bank_latents():
    checkpoint = torch.load(z_bank_path, map_location='cpu')
    z_bank = checkpoint['z_bank'][:sample_count]
    labels = checkpoint.get('labels')

    if labels is not None:
        labels = labels[:sample_count]

    return z_bank.to(device), labels


def tsne_2d(latents):
    return TSNE(
        n_components=2,
        perplexity=30,
        init='pca',
        learning_rate='auto',
        random_state=0,
    ).fit_transform(latents.cpu().numpy())


def plot_latents(encoder_z, lem_z, labels):
    all_z = torch.cat([encoder_z, lem_z], dim=0).cpu()
    projected = tsne_2d(all_z)
    encoder_xy = projected[: encoder_z.size(0)]
    lem_xy = projected[encoder_z.size(0) :]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    panels = (
        (axes[0], encoder_xy, 'VAE encoder latents'),
        (axes[1], lem_xy, 'LEM z_bank latents'),
    )

    for axis, xy, title in panels:
        scatter = axis.scatter(
            xy[:, 0],
            xy[:, 1],
            c=labels,
            cmap='tab10',
            s=32,
            alpha=0.85,
        )
        axis.set_title(title)
        axis.set_xlabel('t-SNE 1')
        axis.set_ylabel('t-SNE 2')
        axis.grid(alpha=0.2)

    fig.colorbar(scatter, ax=axes, ticks=range(10), label='MNIST label')
    fig.suptitle(f'Latent representations for {sample_count} MNIST samples')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')


require_checkpoint(encoder_path)
output_dir.mkdir(exist_ok=True)

images, labels, onehot_labels = load_data()

encoder = Encoder(10, 8).to(device)
encoder.load_state_dict(torch.load(encoder_path, map_location=device))

vae_z = encoder_latents(encoder, images, onehot_labels)

if z_bank_path.exists():
    lem_z, z_bank_labels = load_z_bank_latents()
    if z_bank_labels is not None:
        labels = z_bank_labels
else:
    require_checkpoint(lem_decoder_path)
    lem_decoder = Decoder(8, 10, device=device).to(device)
    lem_decoder.load_state_dict(torch.load(lem_decoder_path, map_location=device))
    lem_z = langevin_latents(lem_decoder, images, onehot_labels)

torch.save(
    {
        'labels': labels,
        'vae_encoder_latents': vae_z.cpu(),
        'lem_z_bank_latents': lem_z.cpu(),
        'it_samples': it_samples,
        'lg_lr': lg_lr,
        'used_saved_z_bank': z_bank_path.exists(),
    },
    latents_path,
)
plot_latents(vae_z, lem_z, labels)

print(f"saved latents to {latents_path}")
print(f"saved plot to {plot_path}")
