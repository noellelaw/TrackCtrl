import matplotlib.pyplot as plt
import torch
from dataset import HurricaneTrackDataset
import numpy as np

def plot_composite_image(composite, prompt):
    """
    Plot the composite RGB image channels separately:
    R = normalized time
    G = reanalysis composite
    B = storm mask
    """
    if isinstance(composite, torch.Tensor):
        composite = composite.cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    channel_names = ['Time (R)', 'Reanalysis (G)', 'Storm Mask (B)']

    for i, ax in enumerate(axs):
        im = ax.imshow(composite[:, :, i], origin='lower', cmap='viridis')
        ax.set_title(channel_names[i])
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Composite Image Visualization\n{prompt}", fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_overlay_mask(composite):
    """
    Overlay storm mask (B) on top of reanalysis (G).
    """
    if isinstance(composite, torch.Tensor):
        composite = composite.cpu().numpy()

    rean = composite[:, :, 1]
    mask = composite[:, :, 2]

    plt.figure(figsize=(6, 5))
    plt.imshow(rean, origin='lower', cmap='gray')
    plt.imshow(mask, origin='lower', cmap='hot', alpha=0.5)
    plt.title("Storm Mask over Reanalysis Composite")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    nc_path = "datasets/era5_storms/era5_storms_combined.nc"
    csv_path = "datasets/era5_storms/hurdat2_north_atlantic.csv"
    
    dataset = HurricaneTrackDataset(csv_path, nc_path)

    idx = 10
    sample = dataset[idx]

    composite = sample['jpg']
    prompt = sample['txt']
    print("Time channel: min", composite[...,0].min(), "max", composite[...,0].max(), "mean", composite[...,0].mean())
    print("Reanalysis channel: min", composite[...,1].min(), "max", composite[...,1].max(), "mean", composite[...,1].mean())
    print("Mask channel: min", composite[...,2].min(), "max", composite[...,2].max(), "mean", composite[...,2].mean())

    print(f"\n=== PLOTTING COMPOSITE IMAGE CHANNELS ===")
    plot_composite_image(composite, prompt)

    print(f"\n=== PLOTTING MASK OVERLAY ===")
    plot_overlay_mask(composite)
