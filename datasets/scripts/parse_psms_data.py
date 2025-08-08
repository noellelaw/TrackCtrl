#!/usr/bin/env python3
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ========= CONFIG =========
BASE_DIR = Path("met_monthly")   # top-level folder
MET_DIR  = BASE_DIR / "data"     # where the .met files live
FILELIST = BASE_DIR / "filelist.txt"
OUT_CSV  = "psmsl_metric_na_2000_2024.csv"

# North Atlantic box
LAT_BOUNDS: Tuple[float, float] = (5.0, 50.0)
LON_BOUNDS: Tuple[float, float] = (-100.0, -10.0)

# Time window
START_YEAR = 2000
END_YEAR   = 2024

# Anomaly threshold for boolean flag
ANOM_Z_THRESHOLD = 2.0

VERBOSE = True
# =========================


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def parse_filelist(filelist_path: Path) -> pd.DataFrame:
    """
    filelist.txt lines look like:
      id; lat; lon; name; coastline; station; Y/N
    For metric, Y/N is not important; values are monthly means in mm relative to station metric datum.
    """
    if not filelist_path.exists():
        raise FileNotFoundError(f"filelist.txt not found at {filelist_path}")

    df = pd.read_csv(
        filelist_path,
        sep=";",
        engine="python",
        header=None,
        names=["station_id", "lat", "lon", "name", "coastline_code", "station_code", "flag"],
    )
    df["station_id"] = df["station_id"].astype(str).str.strip()
    df["name"]       = df["name"].astype(str).str.strip()
    df["lat"]        = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"]        = pd.to_numeric(df["lon"], errors="coerce")
    return df


def filter_na(df: pd.DataFrame) -> pd.DataFrame:
    in_box = (
        df["lat"].between(LAT_BOUNDS[0], LAT_BOUNDS[1], inclusive="both")
        & df["lon"].between(LON_BOUNDS[0], LON_BOUNDS[1], inclusive="both")
    )
    return df[in_box].copy()


def find_station_file(station_id: str) -> Optional[Path]:
    """Find *.met for a station; try multiple paddings + subdirs."""
    sid = int(station_id)
    candidates = [
        MET_DIR  / f"{sid}.metdata",
        MET_DIR/ f"{sid:04d}.metdata",
        MET_DIR / f"{sid:05d}.metdata",
    ]
    for p in candidates:
        if p.exists():
            return p
    for pat in [f"**/{sid}.metdata", f"**/{sid:04d}.metdata", f"**/{sid:05d}.metdata"]:
        matches = list(MET_DIR.glob(pat))
        if matches:
            return matches[0]
    return None


def parse_metric_file(path: Path) -> pd.DataFrame:
    """
    Parse a PSMSL metric monthly file with decimal years like:
      YYYY.FFFF; MSL_mm; flag; code
    """
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";") if p.strip()]
            if len(parts) < 2:
                continue
            try:
                dec_year = float(parts[0])
                msl_mm = float(parts[1])
                if abs(msl_mm) >= 99990:
                    continue  # missing value
                year = int(dec_year)
                month = int(round((dec_year - year) * 12 + 1))
                if month < 1 or month > 12:
                    continue
                rows.append((year, month, msl_mm))
            except Exception:
                continue
    return pd.DataFrame(rows, columns=["year", "month", "msl_mm"])


def build_csv():
    fl = parse_filelist(FILELIST)
    vprint(f"[info] filelist rows: {len(fl)}")
    stations = filter_na(fl)
    vprint(f"[info] NA box {LAT_BOUNDS} x {LON_BOUNDS} → candidate stations: {len(stations)}")

    if stations.empty:
        raise RuntimeError("No stations in bounds; widen LAT_BOUNDS/LON_BOUNDS.")

    all_frames = []
    files_missing = parsed = kept = 0
    year_ranges = []

    for _, r in stations.iterrows():
        sid = r["station_id"]; sname = r["name"]
        lat = float(r["lat"]);  lon = float(r["lon"])

        fpath = find_station_file(sid)
        if fpath is None:
            files_missing += 1
            if files_missing <= 10:
                vprint(f"[warn] missing *.met for station {sid} ({sname})")
            continue

        df = parse_metric_file(fpath)
        parsed += 1
        if df.empty:
            continue

        # Track min/max years for sanity
        year_ranges.append((sid, int(df["year"].min()), int(df["year"].max())))

        # Keep the target window
        df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()
        if df.empty:
            continue

        kept += 1
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
        )
        df["station_id"]   = sid
        df["station_name"] = sname
        df["latitude"]     = lat
        df["longitude"]    = lon

        all_frames.append(df[["station_id","station_name","latitude","longitude","date","msl_mm","month"]])

    vprint(f"[info] summary: candidates={len(stations)}, files_missing={files_missing}, "
           f"parsed_files={parsed}, stations_with_2000_2024={kept}")

    if year_ranges[:5]:
        yr = pd.DataFrame(year_ranges, columns=["station_id","y_min","y_max"])
        vprint("[info] Year coverage (sample):")
        vprint(yr.describe().astype(int))

    if not all_frames:
        raise RuntimeError("No monthly records in 2000–2024. Try widening bounds or years.")

    out = pd.concat(all_frames, ignore_index=True)

    # Anomalies per station per calendar month (climatology over the 2000–2024 window you kept)
    clim = (
        out.groupby(["station_id","month"])["msl_mm"]
           .mean().rename("clim_mm").reset_index()
    )
    stdv = (
        out.groupby(["station_id","month"])["msl_mm"]
           .std(ddof=0).rename("clim_std_mm").reset_index()
    )
    out = out.merge(clim, on=["station_id","month"], how="left") \
             .merge(stdv, on=["station_id","month"], how="left")

    out["anom_mm"] = out["msl_mm"] - out["clim_mm"]
    out["zscore"]  = out["anom_mm"] / out["clim_std_mm"].replace({0.0: np.nan})
    out["zscore"]  = out["zscore"].fillna(0.0)
    out["is_anomaly"] = out["zscore"].abs() >= ANOM_Z_THRESHOLD
    out["source"] = "METRIC"

    out = out[[
        "station_id","station_name","latitude","longitude",
        "date","msl_mm","anom_mm","zscore","is_anomaly","source"
    ]].sort_values(["station_id","date"]).reset_index(drop=True)

    out.to_csv(OUT_CSV, index=False)
    vprint(f"[done] wrote {OUT_CSV} with {len(out):,} rows")


if __name__ == "__main__":
    build_csv()
