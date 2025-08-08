class HurricaneTrackDataset(Dataset):
    def __init__(self, csv_path, nc_path, psmsl_csv,
                 grid_shape=(256, 256), lat_bounds=(5, 45), lon_bounds=(-100, -10),
                 blur_sigma=3, time_offsets_hr=[6, 12, 18, 24, 30], gauge_sigma_px=2.5,
                 use_anomaly=True, debug=False):

        self.df = pd.read_csv(csv_path)
        self.nc = xr.open_dataset(nc_path)

        # --- PSMSL CSV ---
        gauges = pd.read_csv(psmsl_csv, parse_dates=["date"])

        # keep within bounds
        gauges = gauges[
            gauges["latitude"].between(lat_bounds[0], lat_bounds[1]) &
            gauges["longitude"].between(lon_bounds[0], lon_bounds[1])
        ].copy()

        # choose the value column (anomaly or raw)
        value_col = "anom_mm" if use_anomaly and "anom_mm" in gauges.columns else "msl_mm"
        if value_col not in gauges.columns:
            raise ValueError(f"PSMSL CSV must contain 'anom_mm' or 'msl_mm' (got: {list(gauges.columns)})")
        gauges["value"] = gauges[value_col].astype(float)

        # precompute year/month and build lookup by (year, month)
        gauges["year"] = gauges["date"].dt.year
        gauges["month"] = gauges["date"].dt.month
        self.gauges_by_ym = {k: v for k, v in gauges.groupby(["year", "month"])}

        # --- rest unchanged ---
        self.grid_shape = grid_shape
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        self.blur_sigma = blur_sigma
        self.time_offsets_hr = time_offsets_hr
        self.max_time_offset = max(time_offsets_hr)
        self.gauge_sigma_px = gauge_sigma_px
        self.debug = debug

        self.era5_start = pd.to_datetime(str(self.nc.valid_time.min().values))
        self.era5_end   = pd.to_datetime(str(self.nc.valid_time.max().values))

        self.storm_groups = []
        for storm_id, group in self.df.groupby("storm_id"):
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
        m = float(blob.max())
        if m > 1e-8:
            blob /= m
        return blob

    def rasterize_gauges(self, year, month):
        H, W = self.grid_shape
        df = self.gauges_by_ym.get((int(year), int(month)))
        if df is None or df.empty:
            return np.zeros((H, W), dtype=np.float32)

        grid = np.zeros((H, W), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)

        for _, r in df.iterrows():
            i, j = self.latlon_to_grid(float(r["latitude"]), float(r["longitude"]))
            blob = self.make_gaussian_blob(i, j, H, W, self.gauge_sigma_px)
            grid += blob * float(r["value"])
            weight += blob

        m = weight > 1e-8
        grid[m] /= weight[m]
        return grid

    def get_reanalysis_composite(self, date_str, time_str):
        t = str(time_str).zfill(4)
        dt64 = np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}T{t[:2]}:{t[2:]}")
        idx = np.argmin(np.abs(self.nc.valid_time.values - dt64))

        u10 = self.nc.u10.isel(valid_time=idx).interp(latitude=self.lat_grid, longitude=self.lon_grid).values
        v10 = self.nc.v10.isel(valid_time=idx).interp(latitude=self.lat_grid, longitude=self.lon_grid).values
        gauges = self.rasterize_gauges(int(date_str[:4]), int(date_str[4:6]))

        def norm(a):
            a = a.astype(np.float32)
            lo, hi = np.nanpercentile(a, 1), np.nanpercentile(a, 99)
            if hi > lo:
                a = np.clip((a - lo) / (hi - lo), 0, 1)
            else:
                a = np.zeros_like(a, dtype=np.float32)
            return a

        return np.stack([norm(u10), norm(v10), norm(gauges)], axis=-1)
