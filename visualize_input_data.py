import matplotlib.pyplot as plt
import torch
from dataset import HurricaneTrackHeatmapDataset  # Updated dataset class
import numpy as np

def plot_future_track_heatmaps(future_heatmap, conditioning, dataset, prompt, time_offsets_hr):
    """
    Plot non-empty future track heatmaps over MSL background.
    """
    msl = conditioning[0].cpu().numpy()
    T = future_heatmap.shape[0]

    # Collect indices of meaningful heatmaps
    valid_indices = [t for t in range(T) if future_heatmap[t].sum().item() > 1e-6]

    if not valid_indices:
        print("⚠ No valid future heatmaps to plot!")
        return

    fig, axs = plt.subplots(1, len(valid_indices), figsize=(5 * len(valid_indices), 5))

    # If only one valid index, axs is not a list
    if len(valid_indices) == 1:
        axs = [axs]

    for ax, t in zip(axs, valid_indices):
        heat = future_heatmap[t].cpu().numpy()
        ax.imshow(msl, origin='lower', cmap='gray', alpha=0.5)
        im = ax.imshow(heat, origin='lower', cmap='hot', alpha=0.7, vmin=0, vmax=1)
        ax.set_title(f"Future +{time_offsets_hr[t]}h")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Future track heatmaps\n{prompt}", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_reanalysis_channels(reanalysis, prompt, reanalysis_names=['MSL', 'U10', 'V10']):
    """
    Plot reanalysis conditioning fields.
    """
    num_rean = reanalysis.shape[0]
    fig, axs = plt.subplots(1, num_rean, figsize=(5 * num_rean, 5))

    for i in range(num_rean):
        ax = axs[i]
        data = reanalysis[i].cpu().numpy()
        im = ax.imshow(data, origin='lower', cmap='viridis')
        ax.set_title(f"{reanalysis_names[i]}")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Reanalysis fields\n{prompt}", fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    nc_path = "datasets/era5_storms/era5_storms_combined.nc"
    csv_path = "datasets/era5_storms/hurdat2_north_atlantic.csv"
    
    dataset = HurricaneTrackHeatmapDataset(csv_path, nc_path)

    idx = 10
    sample = dataset[idx]

    future_heatmap = sample['target']
    conditioning = sample['conditioning']
    prompt = sample['txt']
    time_offsets_hr = dataset.time_offsets_hr

    # Plot future track heatmaps
    print(f"\n=== PLOTTING FUTURE TRACK HEATMAPS ===")
    plot_future_track_heatmaps(future_heatmap, conditioning, dataset, prompt, time_offsets_hr)

    # Plot reanalysis fields
    print(f"\n=== PLOTTING REANALYSIS FIELDS ===")
    plot_reanalysis_channels(conditioning, prompt)
