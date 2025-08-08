import pandas as pd
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

class HurricaneTrackDataset(Dataset):
    def __init__(self, csv_path, nc_path, psmsl_csv,
                 grid_shape=(256, 256), lat_bounds=(5, 45), lon_bounds=(-100, -10),
                 blur_sigma=3, time_offsets_hr=[6, 12, 18, 24, 30], gauge_sigma_px=2.5,
                 use_anomaly=True, debug=False):
        self.df = pd.read_csv(csv_path)
        self.nc = xr.open_dataset(nc_path)

        self.psmsl_df = pd.read_csv(psmsl_csv, parse_dates=["date"])
        self.psmsl_df = self.psmsl_df[
            self.psmsl_df["latitude"].between(lat_bounds[0], lat_bounds[1]) &
            self.psmsl_df["longitude"].between(lon_bounds[0], lon_bounds[1])
        ].copy()

        if use_anomaly and "is_anomaly" in self.psmsl_df.columns:
            self.psmsl_df["value"] = self.psmsl_df["is_anomaly"]
        elif "value" in self.psmsl_df.columns:
            self.psmsl_df["value"] = self.psmsl_df["value"]
        else:
            raise ValueError("PSMSL CSV must contain 'value' or 'is_anomaly' column.")

        self.grid_shape = grid_shape
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        self.blur_sigma = blur_sigma
        self.time_offsets_hr = time_offsets_hr
        self.max_time_offset = max(time_offsets_hr)
        self.gauge_sigma_px = gauge_sigma_px
        self.debug = debug

        self.era5_start = pd.to_datetime(str(self.nc.valid_time.min().values))
        self.era5_end = pd.to_datetime(str(self.nc.valid_time.max().values))

        storm_groups = list(self.df.groupby("storm_id"))
        self.storm_groups = []
        for storm_id, group in storm_groups:
            group = group.copy()
            group["datetime"] = pd.to_datetime(
                group["date"].astype(str).str.zfill(8) + group["time"].astype(str).str.zfill(4),
                format="%Y%m%d%H%M", errors="coerce"
            )
            if (group["datetime"] >= self.era5_start).any() and (group["datetime"] <= self.era5_end).any():
                self.storm_groups.append((storm_id, group))

        H, W = grid_shape
        self.lat_grid = np.linspace(lat_bounds[0], lat_bounds[1], H)
        self.lon_grid = np.linspace(lon_bounds[0], lon_bounds[1], W)

    def latlon_to_grid(self, lat, lon):
        H, W = self.grid_shape
        i = int((lat - self.lat_bounds[0]) / (self.lat_bounds[1] - self.lat_bounds[0]) * (H - 1))
        j = int((lon - self.lon_bounds[0]) / (self.lon_bounds[1] - self.lon_bounds[0]) * (W - 1))
        return np.clip(i, 0, H - 1), np.clip(j, 0, W - 1)

    def make_gaussian_blob(self, i, j, H, W, sigma):
        y = np.arange(H)[:, None]
        x = np.arange(W)[None, :]
        blob = np.exp(-((x - j)**2 + (y - i)**2) / (2.0 * sigma**2)).astype(np.float32)
        m = blob.max()
        if m > 1e-8:
            blob /= m
        return blob

    def rasterize_gauges(self, year, month):
        H, W = self.grid_shape
        gauges = self.psmsl_df[
            (self.psmsl_df["date"].dt.year == year) &
            (self.psmsl_df["date"].dt.month == month)
        ]
        grid = np.zeros((H, W), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)
        for _, r in gauges.iterrows():
            i, j = self.latlon_to_grid(r["lat"], r["lon"])
            blob = self.make_gaussian_blob(i, j, H, W, self.gauge_sigma_px)
            grid += blob * r["value"]
            weight += blob
        mask = weight > 1e-8
        grid[mask] /= weight[mask]
        return grid

    def get_reanalysis_composite(self, date_str, time_str):
        time_str_padded = time_str.zfill(4)
        dt64 = np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}T{time_str_padded[:2]}:{time_str_padded[2:]}")
        idx = np.argmin(np.abs(self.nc.valid_time.values - dt64))
        u10 = self.nc.u10.isel(valid_time=idx).interp(latitude=self.lat_grid, longitude=self.lon_grid).values
        v10 = self.nc.v10.isel(valid_time=idx).interp(latitude=self.lat_grid, longitude=self.lon_grid).values
        gauges = self.rasterize_gauges(int(date_str[:4]), int(date_str[4:6]))
        def norm(a):
            lo, hi = np.nanpercentile(a, 1), np.nanpercentile(a, 99)
            return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)
        return np.stack([norm(u10), norm(v10), norm(gauges)], axis=-1)

    def __len__(self):
        return len(self.storm_groups)

    def __getitem__(self, idx):
        storm_id, group = self.storm_groups[idx]
        origin_row = group.iloc[0]
        lat0, lon0 = self.parse_latlon(origin_row)
        prompt = f"Given storm origin at {lat0:.2f}, {lon0:.2f}, predict its track."
        hint = torch.from_numpy(self.get_reanalysis_composite(str(origin_row["date"]), str(origin_row["time"])))
        return {"hint": hint, "txt": prompt}
