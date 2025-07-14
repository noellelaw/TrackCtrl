import pandas as pd
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

class HurricaneTrackDataset(Dataset):
    def __init__(self, csv_path, nc_path, grid_shape=(256, 256), lat_bounds=(5, 45), lon_bounds=(-100, -10),
                 blur_sigma=3, time_offsets_hr=[6, 12, 18, 24, 30], debug=False):
        self.df = pd.read_csv(csv_path, header=0)
        self.nc = xr.open_dataset(nc_path)

        self.grid_shape = grid_shape
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        self.blur_sigma = blur_sigma
        self.time_offsets_hr = time_offsets_hr
        self.max_time_offset = max(time_offsets_hr)
        self.debug = debug

        self.known_statuses = ['TD', 'TS', 'HU', 'EX', 'SD', 'SS', 'LO', 'WV', 'DB']

        # ERA5 coverage period (adjust as needed)
        self.era5_start = pd.to_datetime(str(self.nc.valid_time.min().values))
        self.era5_end = pd.to_datetime(str(self.nc.valid_time.max().values))

        # Group storms
        storm_groups = list(self.df.groupby('storm_id'))

        # Filter groups to only storms with at least one point within ERA5
        self.storm_groups = []
        for storm_id, group in storm_groups:
            group = group.copy()
            group['datetime'] = pd.to_datetime(
                group['date'].astype(str).str.zfill(8) + group['time'].astype(str).str.zfill(4),
                format='%Y%m%d%H%M',
                errors='coerce'
            )
            if (group['datetime'] >= self.era5_start).any() and (group['datetime'] <= self.era5_end).any():
                self.storm_groups.append((storm_id, group))

        if debug:
            print(f"Filtered storm groups: {len(self.storm_groups)} storms retained with ERA5 overlap.")

        H, W = grid_shape
        self.lat_grid = np.linspace(lat_bounds[0], lat_bounds[1], H)
        self.lon_grid = np.linspace(lon_bounds[0], lon_bounds[1], W)

    def __len__(self):
        return sum(len(group[1]) for group in self.storm_groups)

    def latlon_to_grid(self, lat, lon):
        H, W = self.grid_shape
        i = int((lat - self.lat_bounds[0]) / (self.lat_bounds[1] - self.lat_bounds[0]) * (H - 1))
        j = int((lon - self.lon_bounds[0]) / (self.lon_bounds[1] - self.lon_bounds[0]) * (W - 1))
        i = np.clip(i, 0, H - 1)
        j = np.clip(j, 0, W - 1)
        return i, j

    def parse_latlon(self, row_values):
        if str(row_values[5]).strip() in self.known_statuses:
            lat_str = str(row_values[6]).strip()
            lon_str = str(row_values[7]).strip()
        else:
            lat_str = str(row_values[5]).strip()
            lon_str = str(row_values[6]).strip()

        lat = float(lat_str[:-1]) * (-1 if lat_str.endswith('S') else 1) if lat_str[-1] in 'NS' else float(lat_str)
        lon = float(lon_str[:-1]) * (-1 if lon_str.endswith('W') else 1) if lon_str[-1] in 'EW' else -float(lon_str)
        return lat, lon

    def get_reanalysis_composite(self, date_str, time_str):
        time_str_padded = time_str.zfill(4)
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        time_formatted = f"{time_str_padded[:2]}:{time_str_padded[2:]}"
        dt64 = np.datetime64(f"{date_formatted}T{time_formatted}")

        idx = np.argmin(np.abs(self.nc.valid_time.values - dt64))

        msl = self.nc.msl.isel(valid_time=idx).interp(latitude=self.lat_grid, longitude=self.lon_grid).values
        u10 = self.nc.u10.isel(valid_time=idx).interp(latitude=self.lat_grid, longitude=self.lon_grid).values
        v10 = self.nc.v10.isel(valid_time=idx).interp(latitude=self.lat_grid, longitude=self.lon_grid).values

        msl_norm = (msl - np.min(msl)) / (np.max(msl) - np.min(msl) + 1e-8)
        u10_norm = (u10 - np.min(u10)) / (np.max(u10) - np.min(u10) + 1e-8)
        v10_norm = (v10 - np.min(v10)) / (np.max(v10) - np.min(v10) + 1e-8)

        composite = np.stack([msl_norm, u10_norm, v10_norm], axis=-1).astype(np.float32)
        return composite

    def make_target_from_latlon(self, lat, lon, sigma=2.5):
        H, W = self.grid_shape
        i, j = self.latlon_to_grid(lat, lon)
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        blob = np.exp(-((x - j) ** 2 + (y - i) ** 2) / (2 * sigma ** 2)).astype(np.float32)
        target = np.stack([blob, blob, blob], axis=-1)
        return target

    def __getitem__(self, idx):
        storm_id, group = self.storm_groups[idx]
        group = group.reset_index(drop=True)

        # Origin row
        origin_row = group.iloc[0]
        lat0, lon0 = self.parse_latlon(origin_row)
        norm_lat0 = (lat0 - self.lat_bounds[0]) / (self.lat_bounds[1] - self.lat_bounds[0])
        norm_lon0 = (lon0 - self.lon_bounds[0]) / (self.lon_bounds[1] - self.lon_bounds[0])

        # Prompt! Construct prompt bc what even was i doing with the crossattention beforehand
        # I can't believe I was feeding the storm name in as if it mattered
        prompt = f"Storm starts at lat={norm_lat0:.3f}, lon={norm_lon0:.3f}. Predict the full trajectory."

        # Input! Reanalysis at origin time
        reanalysis = self.get_reanalysis_composite(str(origin_row['date']), str(origin_row['time']))
        reanalysis = torch.from_numpy(reanalysis).float()  # (H, W, 3)

        # Target! full track, Gaussian blobs at each point technically but we do have 100% certainty
        # that the storm will be at each point in the track, so we can just use the Gaussian
        # to encode the uncertainty in the prediction.
        H, W = self.grid_shape
        target = np.zeros((H, W, 3), dtype=np.float32)
        base_time = pd.to_datetime(str(origin_row['date']) + str(origin_row['time']).zfill(4), format='%Y%m%d%H%M')

        for row in group.itertuples():
            dt = pd.to_datetime(str(row.date) + str(row.time).zfill(4), format='%Y%m%d%H%M')
            norm_t = np.clip((dt - base_time).total_seconds() / 3600.0 / self.max_time_offset, 0, 1)
            lat, lon = self.parse_latlon(row)
            i, j = self.latlon_to_grid(lat, lon)

            blob = self.make_gaussian_blob(i, j, H, W, sigma=2.5)
            target[..., 0] += blob * (lat - self.lat_bounds[0]) / (self.lat_bounds[1] - self.lat_bounds[0])  # lat
            target[..., 1] += blob * (lon - self.lon_bounds[0]) / (self.lon_bounds[1] - self.lon_bounds[0])  # lon
            target[..., 2] += blob * norm_t  # time encoding

        target = torch.from_numpy(target).float()
        
        # Create a dictionary to return reanalysis (H,W,3), prompt, and target (H,W,3)
        # The target is a Gaussian blob centered on the next storm position, but allows
        # us to extract uncertainty in the prediction. Postprocessing will do some argmax thing.
        return {
            "hint": reanalysis,
            "prompt": prompt,
            "target": target
        }
    