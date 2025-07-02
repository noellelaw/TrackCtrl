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
            # Check if any points fall within ERA5 period
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

        # Normalize each separately
        msl_norm = (msl - np.min(msl)) / (np.max(msl) - np.min(msl) + 1e-8)
        u10_norm = (u10 - np.min(u10)) / (np.max(u10) - np.min(u10) + 1e-8)
        v10_norm = (v10 - np.min(v10)) / (np.max(v10) - np.min(v10) + 1e-8)

        # Stack into (H, W, 3)
        composite = np.stack([msl_norm, u10_norm, v10_norm], axis=-1).astype(np.float32)

        return composite

    def __getitem__(self, idx):
        storm_idx = 0
        while idx >= len(self.storm_groups[storm_idx][1]):
            idx -= len(self.storm_groups[storm_idx][1])
            storm_idx += 1

        storm_id, group = self.storm_groups[storm_idx]
        group = group.reset_index(drop=True)
        current_row = group.iloc[idx]

        # Get reanalysis composite (H, W, 3)
        reanalysis = self.get_reanalysis_composite(str(current_row['date']), str(current_row['time']))
        reanalysis = torch.from_numpy(reanalysis).float()  # (H, W, 3)

        H, W = self.grid_shape
        target = np.zeros((H, W, 3), dtype=np.float32)  # (H, W, 3)

        base_datetime = pd.to_datetime(
            str(current_row['date']) + str(current_row['time']).zfill(4),
            format='%Y%m%d%H%M'
        )

        # Loop over track points
        for row_fut in group.itertuples():
            row_values = list(row_fut)
            lat, lon = self.parse_latlon(row_values)

            fut_datetime = pd.to_datetime(
                str(row_fut.date) + str(row_fut.time).zfill(4),
                format='%Y%m%d%H%M'
            )
            t_offset = (fut_datetime - base_datetime).total_seconds() / 3600.0

            i, j = self.latlon_to_grid(lat, lon)

            norm_lat = (lat - self.lat_bounds[0]) / (self.lat_bounds[1] - self.lat_bounds[0])
            norm_lon = (lon - self.lon_bounds[0]) / (self.lon_bounds[1] - self.lon_bounds[0])
            norm_t = np.clip(t_offset / max(self.time_offsets_hr), 0, 1)

            target[i, j, 0] = norm_lat
            target[i, j, 1] = norm_lon
            target[i, j, 2] = norm_t

        target = torch.from_numpy(target).float()

        prompt = (
            f"Storm {str(current_row.storm_name).strip()}, "
            f"status {str(current_row.status) if 'status' in current_row else 'NA'}, "
            f"at {str(current_row.date)} {str(current_row.time).zfill(4)}, "
            f"located at {str(current_row.lat)}, {str(current_row.lon)}."
        )

        return {
            "hint": reanalysis,  # (H, W, 3)
            "jpg": target,       # (H, W, 3)
            "txt": prompt
        }
