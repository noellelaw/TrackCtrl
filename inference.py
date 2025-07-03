import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import HurricaneTrackDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch
import einops
import random
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt

from cldm.ddim_hacked import DDIMSampler
import numpy as np
def run_inference(model, dataloader, ddim_sampler, device, steps=50, strength=1.0, scale=9.0, eta=0.0, guess_mode=False, seed=-1):
    model.eval()
    model.to(device)

    all_predictions = []
    all_inputs = []
    all_targets = []
    all_prompts = []

    with torch.no_grad():
        for batch in dataloader:
            hint = batch['hint']
            target = batch['jpg']  # true track (ground truth)
            prompts = batch['txt']
            num_samples = hint.shape[0]

            # Ensure channels-first
            if hint.ndim == 4 and hint.shape[-1] in [1, 3]:
                hint = einops.rearrange(hint, 'b h w c -> b c h w')

            hint = hint.float().to(device) / 255.0 if hint.max() > 1.0 else hint.float().to(device)

            control = hint.clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            cond = {
                "c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning([p for p in prompts])]
            }
            un_cond = {
                "c_concat": None if guess_mode else [control],
                "c_crossattn": [model.get_learned_conditioning([""] * num_samples)]
            }

            _, _, H, W = control.shape
            latent_shape = (4, H // 8, W // 8)

            model.control_scales = (
                [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else [strength] * 13
            )

            samples, intermediates = ddim_sampler.sample(
                steps, num_samples, latent_shape, cond, device=device, verbose=False, eta=eta,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond
            )

            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5)
            x_samples = x_samples.cpu().numpy().clip(0, 255).astype('uint8')

            all_predictions.append(x_samples)
            all_inputs.append(hint.cpu())
            all_targets.append(target.cpu())
            all_prompts.extend(prompts)

            break  # Remove to process all batches

    all_predictions = np.concatenate(all_predictions, axis=0)
    return all_inputs[0], all_targets[0], all_predictions, all_prompts


def plot_inference_results(inputs, targets, predictions, prompts):
    batch_size = predictions.shape[0]
    for i in range(batch_size):
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # Input hint
        hint_img = einops.rearrange(inputs[i], 'c h w -> h w c').numpy()
        axs[0].imshow(hint_img)
        axs[0].set_title('Input Hint (Reanalysis)')
        axs[0].axis('off')

        # Ground truth
        target_img = targets[i].numpy()
        axs[1].imshow(target_img)
        axs[1].set_title('Expected Track (Ground Truth)')
        axs[1].axis('off')

        # Prediction
        axs[2].imshow(predictions[i])
        axs[2].set_title('Predicted Output')
        axs[2].axis('off')

        plt.suptitle(prompts[i])
        plt.show()

        # Print summary of expected values
        print(f"\nPrompt: {prompts[i]}")
        print(f"Expected (Target) min/max per channel: {[target_img[..., c].min() for c in range(3)]} / {[target_img[..., c].max() for c in range(3)]}")
        print(f"Predicted min/max: {predictions[i].min()} / {predictions[i].max()}")


def main():
    resume_path = 'lightning_logs/version_4/checkpoints/epoch=0-step=719.ckpt'
    batch_size = 4
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False

    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    ddim_sampler = DDIMSampler(model)

    csv_path = 'datasets/era5_storms/hurdat2_north_atlantic_val.csv'
    nc_path = 'datasets/era5_storms/era5_storms_combined_val.nc'
    dataset = HurricaneTrackDataset(csv_path, nc_path)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inputs, targets, predictions, prompts = run_inference(
        model=model,
        dataloader=dataloader,
        ddim_sampler=ddim_sampler,
        device=device,
        steps=500,
        strength=1.0,
        scale=9.0,
        eta=0.0,
        guess_mode=False
    )

    plot_inference_results(inputs, targets, predictions, prompts)

if __name__ == "__main__":
    main()
