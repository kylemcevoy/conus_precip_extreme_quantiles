import numpy as np
import xarray as xr

proj_dir = '/home/data/projects/conus_precip_extremes/'
data_dir = proj_dir + 'obsLE/gpcc_cvdp/'
#data_dir = proj_dir + 'gamma_obsLE/'
output_dir = data_dir + 'analysis/'

# obsLE = xr.open_dataarray(data_dir + 'obsLE.nc')

# obsLE_q95 = obsLE.quantile(0.95, dim='time')
# obsLE_q95.to_netcdf(output_dir + 'obsLE_overall_q95.nc')

# obsLE_month_q95 = (obsLE.groupby('time.month')
#                    .quantile(0.95, dim='time'))
# obsLE_month_q95.to_netcdf(output_dir + 'obsLE_monthly_q95.nc')

# gpcc = xr.open_dataarray(proj_dir + 'gpcc/gpcc_mmday.nc')

# gpcc_q95 = gpcc.quantile(0.95, dim='time')
# gpcc_q95.to_netcdf(output_dir + 'gpcc_overall_q95.nc')

# gpcc_month_q95 = gpcc.groupby('time.month').quantile(0.95, dim='time')
# gpcc_month_q95.to_netcdf(output_dir + 'gpcc_monthly_q95.nc')

# obsLE_totals = obsLE * obsLE.time.dt.days_in_month
# obsLE_totals.to_netcdf(data_dir + 'obsLE_totals.nc')

# chunk_mems = [np.arange((j * 100), (j + 1) * 100) for j in range(10)]

# chunk_quantile_list = []
# for chunk in chunk_mems:
#     file_names = [f'{data_dir}obsLE_member{mem:04}.nc' for mem in chunk]
#     chunk_data = xr.open_mfdataset(file_names,
#                                    combine='nested',
#                                    concat_dim='ens_mem')
#     chunk_data = chunk_data.assign_coords({'ens_mem': chunk + 1})
#     chunk_quantiles = chunk_data.groupby('time.month').quantile(0.95, dim='time')
#     chunk_quantile_list.append(chunk_quantiles)
    
# obsLE_q95 = xr.concat(chunk_quantile_list, dim='ens_mem')
# obsLE_q95.to_netcdf(output_dir + 'obsLE_monthly_total_q95.nc')

# obsLE_totals = xr.open_dataarray(data_dir + 'obsLE_totals.nc')
# obsLE_seasonal_totals = obsLE_totals.resample({'time': 'QS-DEC'}).sum('time')
# obsLE_seasonal_totals = 

# obsLE_total_q95 = obsLE_totals.quantile(0.95, dim='time')
# obsLE_total_q95.to_netcdf(output_dir + 'totals_q95/obsLE_total_q95.nc')

# obsLE_monthly_total_q95 = obsLE_totals.groupby('time.month').quantile(0.95, dim='time')
# obsLE_monthly_total_q95.to_netcdf(output_dir + 'totals_q95/obsLE_total_monthly_q95.nc')

# obsLE_monthly_total_q05 = obsLE_totals.groupby('time.month').quantile(0.05, dim='time')
# obsLE_monthly_total_q05.to_netcdf(output_dir + 'totals_q95/obsLE_total_monthly_q05.nc')

# gpcc_total = xr.open_dataarray(proj_dir + 'gpcc/gpcc_totals.nc')
# gpcc_total_q95 = gpcc_total.quantile(0.95, dim='time')
# gpcc_total_q05 = gpcc_total.quantile(0.05, dim='time')

# gpcc_total_q95.to_netcdf(output_dir + 'totals_q95/gpcc_total_q95.nc')
# gpcc_total_q05.to_netcdf(output_dir + 'totals_q05/gpcc_total_q05.nc')

# gpcc_total_monthly_q95 = (gpcc_total.groupby('time.month')
#                           .quantile(0.95, dim='time'))
# gpcc_total_monthly_q95.to_netcdf(output_dir + 
#                                  'totals_q05/gpcc_total_monthly_q95.nc')

# gpcc_total_monthly_q05 = (gpcc_total.groupby('time.month')
#                           .quantile(0.05, dim='time'))
# gpcc_total_monthly_q05.to_netcdf(output_dir + 
#                                  'totals_q05/gpcc_total_monthly_q05.nc')

chunk_mems = [np.arange((j * 100) + 1, (j + 1) * 100) + 1 for j in range(10)]

chunk_q95_list = []
chunk_q05_list = []
for i, chunk in enumerate(chunk_mems):
    file_names = [f'{data_dir}obsLE_member{mem:04}.nc' for mem in chunk]
    chunk_data = xr.open_mfdataset(file_names,
                                   combine='nested',
                                   concat_dim='ens_mem')
    chunk_data = chunk_data.assign_coords({'ens_mem': chunk})
    chunk_totals = chunk_data * chunk_data.time.dt.days_in_month
    chunk_data_seasonal = chunk_totals.resample({'time': 'QS-DEC'}).sum('time')
    chunk_data_djf = chunk_data_seasonal.sel(
        time=chunk_data_seasonal['time.month'] == 12)
    chunk_djf_q95 = chunk_data_djf.quantile(0.95, dim='time')
    chunk_djf_q05 = chunk_data_djf.quantile(0.05, dim='time')
    chunk_djf_q95.to_netcdf(output_dir + f'obsLE_djf_q95_chunk{i}.nc')
    chunk_djf_q05.to_netcdf(output_dir + f'obsLE_djf_q05_chunk{i}.nc')
    chunk_q95_list.append(chunk_djf_q95)
    chunk_q05_list.append(chunk_djf_q05)
    
obsLE_djf_q95 = xr.concat(chunk_q95_list, dim='ens_mem')
obsLE_djf_q95.to_netcdf(output_dir + 'obsLE_djf_q95.nc')

obsLE_djf_q05 = xr.concat(chunk_q05_list, dim='ens_mem')
obsLE_djf_q05.to_netcdf(output_dir + 'obsLE_djf_q05.nc')

gpcc_total = xr.open_dataarray(proj_dir + 'gpcc/gpcc_totals.nc')
gpcc_seasons = gpcc_total.resample({'time': 'QS-DEC'}).sum('time')
gpcc_djf = gpcc_seasons.sel(time=gpcc_seasons['time.month'] == 12)
gpcc_djf_q95 = gpcc_djf.quantile(0.95, dim='time')
gpcc_djf_q05 = gpcc_djf.quantile(0.05, dim='time')

gpcc_djf_q95.to_netcdf(output_dir + 'gpcc_djf_q95.nc')
gpcc_djf_q05.to_netcdf(output_dir + 'gpcc_djf_q05.nc')