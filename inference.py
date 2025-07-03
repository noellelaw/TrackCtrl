import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import HurricaneTrackDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

def plot_inference_results(inputs, predictions, prompts):
    """
    Plot inputs, predictions side by side.
    inputs: dict of model inputs, e.g., 'hint', 'jpg'
    predictions: model outputs (e.g., denoised or sampled image)
    prompts: list of prompt strings
    """
    batch_size = predictions.shape[0]
    for i in range(batch_size):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Input conditioning (hint)
        hint_img = inputs['hint'][i].cpu().permute(1, 2, 0).numpy()
        axs[0].imshow(hint_img)
        axs[0].set_title('Input Hint (Reanalysis)')
        axs[0].axis('off')

        # Target (true track)
        target_img = inputs['jpg'][i].cpu().permute(1, 2, 0).numpy()
        axs[1].imshow(target_img)
        axs[1].set_title('Target Track (True)')
        axs[1].axis('off')

        # Prediction
        pred_img = predictions[i].cpu().permute(1, 2, 0).numpy()
        axs[2].imshow(pred_img)
        axs[2].set_title('Predicted Track (Output)')
        axs[2].axis('off')

        plt.suptitle(prompts[i])
        plt.show()

def run_inference(model, dataloader, device):
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_inputs = []
    all_prompts = []

    with torch.no_grad():
        for batch in dataloader:
            # Move inputs to device
            hint = batch['hint'].to(device)

            # Forward pass
            # Assuming model.forward returns output sample image
            output = model.sample(hint=hint, cond=batch['txt'])  # adjust if needed

            # Collect results
            all_predictions.append(output)
            all_inputs.append(batch)
            all_prompts.extend(batch['txt'])

            # For demo: break after one batch
            break

    # Concatenate results
    all_predictions = torch.cat(all_predictions, dim=0)
    return all_inputs[0], all_predictions, all_prompts

def main():
    resume_path = 'epochX.pth'
    batch_size = 4
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False

    # Create model
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    csv_path = 'datasets/era5_storms/hurdat2_north_atlantic_val.csv'
    nc_path = 'datasets/era5_storms/era5_storms_combined_val.nc'
    dataset = HurricaneTrackDataset(csv_path, nc_path)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run inference
    inputs, predictions, prompts = run_inference(model, dataloader, device)

    # Plot results
    plot_inference_results(inputs, predictions, prompts)

if __name__ == "__main__":
    main()
