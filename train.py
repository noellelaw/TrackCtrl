from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import HurricaneTrackHeatmapDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# Configs
resume_path = './models/control_sd21_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# Create model
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Create dataset + dataloader
csv_path = 'datasets/era5_storms/hurdat2_north_atlantic.csv'
nc_path = 'datasets/era5_storms/era5_storms_combined.nc'
dataset = HurricaneTrackHeatmapDataset(csv_path, nc_path)
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)

# Logger
logger = ImageLogger(batch_frequency=logger_freq)

# Trainer
trainer = pl.Trainer(gpus=0, precision=32, callbacks=[logger])

# Train
trainer.fit(model, dataloader)