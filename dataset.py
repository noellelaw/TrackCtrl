import pandas as pd
import numpy as np
import torch
import xarray as xr
import cv2
from torch.utils.data import Dataset

class HurricaneTrackHeatmapDataset(Dataset):
    def __init__(self, csv_path, nc_path, grid_shape=(256, 256), lat_bounds=(5, 45), lon_bounds=(-100, -10),
                 blur_sigma=3, time_offsets_hr=[6, 12, 18, 24, 30], debug=False):
        self.df = pd.read_csv(csv_path, header=0)
        self.nc = xr.open_dataset(nc_path)

        self.grid_shape = grid_shape
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        self.blur_sigma = blur_sigma
        self.time_offsets_hr = time_offsets_hr
        self.debug = debug

        self.storm_groups = list(self.df.groupby('storm_id'))

        H, W = grid_shape
        self.lat_grid = np.linspace(lat_bounds[0], lat_bounds[1], H)
        self.lon_grid = np.linspace(lon_bounds[0], lon_bounds[1], W)

        self.known_statuses = ['TD', 'TS', 'HU', 'EX', 'SD', 'SS', 'LO', 'WV', 'DB']

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

    def get_reanalysis_fields(self, date_str, time_str):
        time_str_padded = time_str.zfill(4)
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        time_formatted = f"{time_str_padded[:2]}:{time_str_padded[2:]}"
        dt64 = np.datetime64(f"{date_formatted}T{time_formatted}")

        idx = np.argmin(np.abs(self.nc.valid_time.values - dt64))

        msl = self.nc.msl.isel(valid_time=idx).interp(latitude=self.lat_grid, longitude=self.lon_grid).values
        u10 = self.nc.u10.isel(valid_time=idx).interp(latitude=self.lat_grid, longitude=self.lon_grid).values
        v10 = self.nc.v10.isel(valid_time=idx).interp(latitude=self.lat_grid, longitude=self.lon_grid).values

        return np.stack([msl, u10, v10], axis=-1).astype(np.float32)  # (H, W, 3)

    def __getitem__(self, idx):
        storm_idx = 0
        while idx >= len(self.storm_groups[storm_idx][1]):
            idx -= len(self.storm_groups[storm_idx][1])
            storm_idx += 1

        storm_id, group = self.storm_groups[storm_idx]
        group = group.reset_index(drop=True)
        current_row = group.iloc[idx]

        reanalysis = self.get_reanalysis_fields(str(current_row['date']), str(current_row['time']))

        base_datetime = pd.to_datetime(
            str(current_row['date']) + str(current_row['time']).zfill(4),
            format='%Y%m%d%H%M'
        )

        T, H, W = len(self.time_offsets_hr), *self.grid_shape
        heatmap = np.zeros((T, H, W), dtype=np.float32)

        for row_fut in group.iloc[(idx+1):].itertuples():
            row_values = list(row_fut)
            lat, lon = self.parse_latlon(row_values)

            fut_datetime = pd.to_datetime(
                str(row_fut.date) + str(row_fut.time).zfill(4),
                format='%Y%m%d%H%M'
            )
            t_offset = (fut_datetime - base_datetime).total_seconds() / 3600.0

            t_idx = np.argmin(np.abs(np.array(self.time_offsets_hr) - t_offset))
            if np.abs(self.time_offsets_hr[t_idx] - t_offset) > 3:
                continue

            i, j = self.latlon_to_grid(lat, lon)
            heatmap[t_idx, i, j] = 1.0

        # Process heatmap
        for t in range(T):
            if self.blur_sigma > 0:
                heatmap[t] = cv2.GaussianBlur(heatmap[t], (0, 0), self.blur_sigma)
            max_val = heatmap[t].max()
            if max_val > 0:
                heatmap[t] /= max_val
            if self.debug:
                print(f"Heatmap +{self.time_offsets_hr[t]}h: min={heatmap[t].min()}, max={heatmap[t].max()}, sum={heatmap[t].sum()}")

        # Convert heatmap sum to (H, W, 3) + normalize
        jpg = heatmap.sum(axis=0)
        jpg = np.clip(jpg, 0, 1)
        jpg = (jpg[..., None] * np.ones(3)).astype(np.float32)  # (H, W, 3)

        # Reanalysis normalize [0,1]
        reanalysis_min = reanalysis.min()
        reanalysis_max = reanalysis.max()
        if reanalysis_max > reanalysis_min:
            hint = (reanalysis - reanalysis_min) / (reanalysis_max - reanalysis_min)
        else:
            hint = reanalysis

        prompt = (
            f"Storm {str(current_row.storm_name).strip()}, "
            f"status {str(current_row.status) if 'status' in current_row else 'NA'}, "
            f"at {str(current_row.date)} {str(current_row.time).zfill(4)}, "
            f"located at {str(current_row.lat)}, {str(current_row.lon)}."
        )

        return {
            "jpg": jpg,     # (H, W, 3), float32, [0,1]
            "hint": hint,   # (H, W, 3), float32, [0,1]
            "txt": prompt
        }
