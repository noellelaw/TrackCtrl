import matplotlib.pyplot as plt
import torch
from dataset import HurricaneTrackReanalysisDataset  
import numpy as np

def plot_future_track_points(future_track, conditioning, dataset, prompt):
    """
    Plot future track lat/lon points over reanalysis MSL field for context.
    """
    # Extract MSL for background
    msl = conditioning[0].cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(msl, origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Future Track over MSL field")

    # Plot track points
    lat_grid = dataset.lat_grid
    lon_grid = dataset.lon_grid

    for point in future_track:
        lat, lon, t_offset = point.tolist()
        # Map lat/lon to pixel space
        i = int((lat - dataset.lat_bounds[0]) / (dataset.lat_bounds[1] - dataset.lat_bounds[0]) * (dataset.grid_shape[0] - 1))
        j = int((lon - dataset.lon_bounds[0]) / (dataset.lon_bounds[1] - dataset.lon_bounds[0]) * (dataset.grid_shape[1] - 1))
        i = np.clip(i, 0, dataset.grid_shape[0] - 1)
        j = np.clip(j, 0, dataset.grid_shape[1] - 1)

        ax.plot(j, i, 'ro')  # red dot
        ax.text(j + 2, i + 2, f"{t_offset:.1f}h", color='white', fontsize=8)

    plt.suptitle(prompt, fontsize=10)
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
    
    dataset = HurricaneTrackReanalysisDataset(csv_path, nc_path)

    idx = 10
    sample = dataset[idx]

    future_track = sample['future_track']
    conditioning = sample['conditioning']
    prompt = sample['txt']

    # Plot future track over MSL
    print(f"\n=== PLOTTING FUTURE TRACK ===")
    plot_future_track_points(future_track, conditioning, dataset, prompt)

    # Plot reanalysis fields
    print(f"\n=== PLOTTING REANALYSIS FIELDS ===")
    plot_reanalysis_channels(conditioning, prompt)
