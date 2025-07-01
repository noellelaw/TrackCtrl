import pandas as pd
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

class HurricaneTrackReanalysisDataset(Dataset):
    def __init__(self, csv_path, nc_path, grid_shape=(256, 256), lat_bounds=(5, 45), lon_bounds=(-100, -10)):
        self.df = pd.read_csv(csv_path, header=0)
        self.nc = xr.open_dataset(nc_path)

        self.grid_shape = grid_shape
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds

        self.storm_groups = list(self.df.groupby('storm_id'))
        
        H, W = grid_shape
        self.lat_grid = np.linspace(lat_bounds[0], lat_bounds[1], H)
        self.lon_grid = np.linspace(lon_bounds[0], lon_bounds[1], W)

    def __len__(self):
        return sum(len(group[1]) for group in self.storm_groups)

    def get_reanalysis_fields(self, date_str, time_str):
        time_str_padded = time_str.zfill(4)
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        time_formatted = f"{time_str_padded[:2]}:{time_str_padded[2:]}"
        datetime_str = f"{date_formatted}T{time_formatted}"
        dt64 = np.datetime64(datetime_str)

        idx = np.argmin(np.abs(self.nc.valid_time.values - dt64))

        msl = self.nc.msl.isel(valid_time=idx).interp(latitude=self.lat_grid, longitude=self.lon_grid).values
        u10 = self.nc.u10.isel(valid_time=idx).interp(latitude=self.lat_grid, longitude=self.lon_grid).values
        v10 = self.nc.v10.isel(valid_time=idx).interp(latitude=self.lat_grid, longitude=self.lon_grid).values

        return np.stack([msl, u10, v10], axis=0).astype(np.float32)

    def __getitem__(self, idx):
        storm_idx = 0
        while idx >= len(self.storm_groups[storm_idx][1]):
            idx -= len(self.storm_groups[storm_idx][1])
            storm_idx += 1

        storm_id, group = self.storm_groups[storm_idx]
        group = group.reset_index(drop=True)
        current_row = group.iloc[idx]

        # Reanalysis at current time
        reanalysis = self.get_reanalysis_fields(str(current_row['date']), str(current_row['time']))
        reanalysis = torch.from_numpy(reanalysis)

        # Prepare future track lat, lon, time_offset
        base_date = str(current_row['date'])
        base_time = str(current_row['time']).zfill(4)
        base_datetime = pd.to_datetime(base_date + base_time, format='%Y%m%d%H%M')

        known_statuses = ['TD', 'TS', 'HU', 'EX', 'SD', 'SS', 'LO', 'WV', 'DB']
        future_track = []

        for idx_fut in range(idx + 1, len(group)):
            row_fut = group.iloc[idx_fut]
            row_values = row_fut.tolist()

            if str(row_values[4]).strip() in known_statuses:
                lat_str = str(row_values[5]).strip()
                lon_str = str(row_values[6]).strip()
            else:
                lat_str = str(row_values[4]).strip()
                lon_str = str(row_values[5]).strip()

            if lat_str.endswith('N'):
                lat = float(lat_str[:-1])
            elif lat_str.endswith('S'):
                lat = -float(lat_str[:-1])
            else:
                lat = float(lat_str)

            if lon_str.endswith('W'):
                lon = -float(lon_str[:-1])
            elif lon_str.endswith('E'):
                lon = float(lon_str[:-1])
            else:
                lon = -float(lon_str)

            fut_date = str(row_fut['date'])
            fut_time = str(row_fut['time']).zfill(4)
            fut_datetime = pd.to_datetime(fut_date + fut_time, format='%Y%m%d%H%M')
            time_offset_hr = (fut_datetime - base_datetime).total_seconds() / 3600.0

            future_track.append([lat, lon, time_offset_hr])

        future_track = torch.tensor(future_track, dtype=torch.float32)

        # Build prompt
        storm_name = str(current_row['storm_name']).strip()
        date = str(current_row['date'])
        time = str(current_row['time']).zfill(4)
        status = str(current_row['status']) if 'status' in current_row else 'NA'
        lat_str = str(current_row['lat'])
        lon_str = str(current_row['lon'])

        prompt = (
            f"Storm {storm_name}, status {status}, at {date} {time}, "
            f"located at {lat_str}, {lon_str}."
        )

        return dict(
            conditioning=reanalysis,  # (3, H, W)
            future_track=future_track,  # (N, 3): lat, lon, time offset
            txt=prompt
        )
