from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from models.vdecoder import Decoder


device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = Path('output')

vae_decoder_path = output_dir / 'vae-decoder.pt'
lem_decoder_path = output_dir / 'lem-decoder.pt'
results_path = output_dir / 'likelihood-results.pt'
table_path = output_dir / 'likelihood-results.md'
plot_path = output_dir / 'likelihood-results.png'

latent_dim = 8
class_count = 10
image_count = 50
latent_samples = 10000
latent_batch_size = 1000


def require_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the training script that creates this checkpoint first."
        )


def load_test_data():
    mnist_data = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    images = mnist_data.data[:image_count].float().flatten(start_dim=1) / 255.0
    labels = mnist_data.targets[:image_count]
    onehot_labels = F.one_hot(labels, num_classes=class_count).float()

    return images.to(device), labels, onehot_labels.to(device)


def load_decoder(path):
    decoder = Decoder(latent_dim, class_count, device=device).to(device)
    decoder.load_state_dict(torch.load(path, map_location=device))
    decoder.eval()
    return decoder


def gaussian_log_prob(x, mean, logvar):
    return -0.5 * (
        torch.log(torch.tensor(2.0 * torch.pi, device=device))
        + logvar
        + (x - mean) ** 2 / torch.exp(logvar)
    ).sum(dim=1)


def estimate_log_likelihood(decoder, images, onehot_labels):
    all_log_probs = []

    with torch.no_grad():
        for start in range(0, latent_samples, latent_batch_size):
            batch_size = min(latent_batch_size, latent_samples - start)
            z = torch.randn(batch_size, latent_dim, device=device)
            z = z.repeat_interleave(images.size(0), dim=0)
            repeated_images = images.repeat(batch_size, 1)
            repeated_labels = onehot_labels.repeat(batch_size, 1)

            means, logvars = decoder(z, repeated_labels)
            log_probs = gaussian_log_prob(repeated_images, means, logvars)
            log_probs = log_probs.view(batch_size, images.size(0))
            all_log_probs.append(log_probs.cpu())

    log_probs = torch.cat(all_log_probs, dim=0)
    return torch.logsumexp(log_probs, dim=0) - torch.log(torch.tensor(float(latent_samples)))


def write_table(labels, vae_ll, lem_ll):
    mean_row = (
        f"| mean | - | {vae_ll.mean().item():.4f} | "
        f"{lem_ll.mean().item():.4f} | {(lem_ll.mean() - vae_ll.mean()).item():.4f} |\n"
    )
    lines = [
        "| image | label | VAE log p(x) | LEM log p(x) | LEM - VAE |",
        "|---:|---:|---:|---:|---:|",
    ]

    for index, label in enumerate(labels):
        diff = lem_ll[index] - vae_ll[index]
        lines.append(
            f"| {index} | {label.item()} | {vae_ll[index].item():.4f} | "
            f"{lem_ll[index].item():.4f} | {diff.item():.4f} |"
        )

    lines.append(mean_row)
    table_path.write_text('\n'.join(lines))


def plot_likelihoods(labels, vae_ll, lem_ll):
    indices = torch.arange(labels.size(0))
    diff = lem_ll - vae_ll

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(indices, vae_ll, marker='o', linewidth=1.5, label='VAE')
    axes[0].plot(indices, lem_ll, marker='o', linewidth=1.5, label='LEM')
    axes[0].set_ylabel('estimated log p(x)')
    axes[0].set_title(f'Monte Carlo test likelihoods, {latent_samples} z samples')
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    bar_colors = ['#2f7d32' if value >= 0 else '#b3261e' for value in diff.tolist()]
    axes[1].bar(indices, diff, color=bar_colors)
    axes[1].axhline(0.0, color='black', linewidth=1)
    axes[1].set_xlabel('test image index')
    axes[1].set_ylabel('LEM - VAE')
    axes[1].grid(axis='y', alpha=0.25)

    label_text = [str(label.item()) for label in labels]
    axes[1].set_xticks(indices)
    axes[1].set_xticklabels(label_text, rotation=0)
    axes[1].text(
        0.01,
        0.95,
        'x-axis labels are MNIST digits',
        transform=axes[1].transAxes,
        va='top',
    )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')


def main():
    require_checkpoint(vae_decoder_path)
    require_checkpoint(lem_decoder_path)
    output_dir.mkdir(exist_ok=True)

    images, labels, onehot_labels = load_test_data()
    vae_decoder = load_decoder(vae_decoder_path)
    lem_decoder = load_decoder(lem_decoder_path)

    vae_ll = estimate_log_likelihood(vae_decoder, images, onehot_labels)
    lem_ll = estimate_log_likelihood(lem_decoder, images, onehot_labels)

    torch.save(
        {
            'labels': labels,
            'vae_log_likelihood': vae_ll,
            'lem_log_likelihood': lem_ll,
            'image_count': image_count,
            'latent_samples': latent_samples,
            'latent_batch_size': latent_batch_size,
        },
        results_path,
    )
    write_table(labels, vae_ll, lem_ll)
    plot_likelihoods(labels, vae_ll, lem_ll)

    print(f"VAE mean log p(x): {vae_ll.mean().item():.4f}")
    print(f"LEM mean log p(x): {lem_ll.mean().item():.4f}")
    print(f"LEM - VAE: {(lem_ll.mean() - vae_ll.mean()).item():.4f}")
    print(f"saved results to {results_path}")
    print(f"saved table to {table_path}")
    print(f"saved plot to {plot_path}")


if __name__ == '__main__':
    main()
