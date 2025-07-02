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

        wind_mag = np.sqrt(u10**2 + v10**2)
        msl_norm = (msl - msl.min()) / (msl.max() - msl.min() + 1e-8)
        wind_norm = (wind_mag - wind_mag.min()) / (wind_mag.max() - wind_mag.min() + 1e-8)

        composite = 0.5 * (msl_norm + wind_norm)
   
        return composite

    def __getitem__(self, idx):
        storm_idx = 0
        while idx >= len(self.storm_groups[storm_idx][1]):
            idx -= len(self.storm_groups[storm_idx][1])
            storm_idx += 1

        storm_id, group = self.storm_groups[storm_idx]
        group = group.reset_index(drop=True)
        current_row = group.iloc[idx]

        base_datetime = pd.to_datetime(
            str(current_row['date']) + str(current_row['time']).zfill(4),
            format='%Y%m%d%H%M'
        )

        H, W = self.grid_shape
        composite_img = np.zeros((H, W, 3), dtype=np.float32)

        # Reanalysis composite (same everywhere in image)
        rean_comp = self.get_reanalysis_composite(str(current_row['date']), str(current_row['time']))

        # Build composite
        for row_fut in group.itertuples():
            row_values = list(row_fut)
            lat, lon = self.parse_latlon(row_values)

            fut_datetime = pd.to_datetime(
                str(row_fut.date) + str(row_fut.time).zfill(4),
                format='%Y%m%d%H%M'
            )
            t_offset = (fut_datetime - base_datetime).total_seconds() / 3600.0
            t_norm = np.clip(t_offset / self.max_time_offset, 0, 1)

            i, j = self.latlon_to_grid(lat, lon)

            composite_img[i, j, 0] = t_norm
            composite_img[i, j, 1] = rean_comp[i, j]
            composite_img[i, j, 2] = 1.0

        prompt = (
            f"Storm {str(current_row.storm_name).strip()}, "
            f"status {str(current_row.status) if 'status' in current_row else 'NA'}, "
            f"at {str(current_row.date)} {str(current_row.time).zfill(4)}, "
            f"located at {str(current_row.lat)}, {str(current_row.lon)}."
        )
                # Check ERA5 valid time vs your storm base time
        print("Storm base datetime:", base_datetime)
        print("Selected ERA5 valid_time:", self.nc.valid_time.values[idx])

        # Check lat/lon grid overlap
        print(f"ERA5 lat: {self.nc.latitude.min().values} to {self.nc.latitude.max().values}")
        print(f"ERA5 lon: {self.nc.longitude.min().values} to {self.nc.longitude.max().values}")
        print(f"My grid lat: {self.lat_grid.min()} to {self.lat_grid.max()}")
        print(f"My grid lon: {self.lon_grid.min()} to {self.lon_grid.max()}")

        # Quick reanalysis stats BEFORE normalization
        print(f"Reanalysis raw min: {rean_comp.min()}, max: {rean_comp.max()}, mean: {rean_comp.mean()}")
 
        return {
            "jpg": composite_img.astype(np.float32),  # (H, W, 3), float32, [0,1]
            "txt": prompt
        }
