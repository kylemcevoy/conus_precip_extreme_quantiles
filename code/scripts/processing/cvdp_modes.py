import xarray as xr

proj_dir = '/home/data/projects/conus_precip_extremes/'
cvdp_dir = proj_dir + 'cvdp/'
output_dir = proj_dir + 'climate_modes/'

cvdp_era5_modes = xr.open_dataset(
    cvdp_dir + 'ERA20c_ERA5_comb.cvdp_data.1920-2020.nc',
    decode_times=False)

cvdp_ersst_modes = xr.open_dataset(
    cvdp_dir + 'ERSSTv5.cvdp_data.1920-2020.nc', 
    decode_times=False)

enso_cvdp = cvdp_ersst_modes['nino34']
pdo_cvdp = cvdp_ersst_modes['pdv_timeseries_mon']
pna_cvdp = cvdp_era5_modes['pna_timeseries_mon']
nao_cvdp = cvdp_era5_modes['nao_timeseries_mon']

time_fixed = xr.cftime_range('1920-01-01', 
                             '2020-12-01', 
                             freq='MS',
                             calendar='standard')

modes_ds = xr.Dataset({'enso': enso_cvdp,
                       'pdo': pdo_cvdp,
                       'pna': pna_cvdp,
                       'nao': nao_cvdp})

modes_ds = modes_ds.assign_coords({'time': time_fixed})

modes_ds.to_netcdf(output_dir + 'cvdp_obs_modes.nc')
