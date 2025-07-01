from share import *
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader
from dataset import HurricaneTrackHeatmapDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    resume_path = './models/control_sd15_seg.pth'
    batch_size = 4
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False

    # Create model on CPU
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    csv_path = 'datasets/era5_storms/hurdat2_north_atlantic.csv'
    nc_path = 'datasets/era5_storms/era5_storms_combined.nc'
    dataset = HurricaneTrackHeatmapDataset(csv_path, nc_path)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

    logger = ImageLogger(batch_frequency=logger_freq)

    # Force CPU usage: set accelerator explicitly
    trainer = pl.Trainer(
        accelerator="cpu",  # ensures CPU
        devices=1,          # one CPU device
        precision=32,
        callbacks=[logger]
    )

    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()
