import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_reanalysis_fields(reanalysis, reanalysis_names=['MSL', 'U10', 'V10']):
    """
    Plot each channel of the raw reanalysis tensor.
    Assumes reanalysis shape: (3, H, W)
    """
    if isinstance(reanalysis, torch.Tensor):
        reanalysis = reanalysis.cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(3):
        ax = axs[i]
        im = ax.imshow(reanalysis[i], origin='lower', cmap='viridis')
        ax.set_title(reanalysis_names[i])
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("ERA5 Reanalysis Fields (Raw)", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_composite_field(composite):
    """
    Plot composite field (H, W)
    """
    if isinstance(composite, torch.Tensor):
        composite = composite.cpu().numpy()

    if composite.ndim == 3:
        # Collapse channels if mistakenly passed (3,H,W)
        composite = composite.mean(axis=0)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(composite, origin='lower', cmap='plasma')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Composite Conditioning Field")
    plt.axis('off')
    plt.show()

def plot_overlay_mask(mask, reanalysis_component, component_name='MSL'):
    """
    Overlay mask on reanalysis field.
    mask: (H,W)
    reanalysis_component: (H,W)
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if isinstance(reanalysis_component, torch.Tensor):
        reanalysis_component = reanalysis_component.cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(reanalysis_component, origin='lower', cmap='gray')
    plt.imshow(mask, origin='lower', cmap='hot', alpha=0.5)
    plt.title(f"Storm Mask over {component_name}")
    plt.colorbar()
    plt.show()

# Example usage in your script
if __name__ == "__main__":
    from dataset import HurricaneTrackDataset

    nc_path = "datasets/era5_storms/era5_storms_combined.nc"
    csv_path = "datasets/era5_storms/hurdat2_north_atlantic.csv"
    
    dataset = HurricaneTrackDataset(csv_path, nc_path, debug=True)
    sample = dataset[10]

    reanalysis = sample['hint']  # (3, H, W)
    target = sample['jpg']       # (3, H, W)
    prompt = sample['txt']

    # Plot raw reanalysis fields
    plot_reanalysis_fields(reanalysis)

    # Plot composite field (average for demo purposes)
    composite_field = reanalysis.mean(axis=0)
    plot_composite_field(composite_field)

    # Plot overlay of mask (using time as mask for demo)
    mask = target[2]  # Assume channel 2 is time
    plot_overlay_mask(mask, reanalysis[0], component_name="MSL")

    print(f"Prompt: {prompt}")
