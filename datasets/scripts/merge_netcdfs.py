import xarray as xr

# Open and combine your NetCDF files
ds = xr.open_mfdataset('era5_storms/20*.nc', combine='by_coords')
# print(ds)

# Save combined dataset to a new NetCDF file
ds.to_netcdf('era5_storms_combined.nc')

print("Saved combined dataset to era5_storms_combined.nc")