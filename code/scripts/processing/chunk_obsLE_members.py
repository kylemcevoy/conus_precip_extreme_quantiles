import numpy as np
import xarray as xr

data_dir = '/home/data/projects/conus_precip_extremes/obsLE/gpcc_cvdp/'

path_lists = []
for j in range(10):
    path_lists.append([data_dir + f'obsLE_member{i:04}.nc' 
                       for i in np.arange(1 + (j * 100), 101 + (j * 100))])

for j, path_list in enumerate(path_lists):
    print(j)
    start_mem = 1 + (j * 100)
    end_mem = 101 + (j * 100)
    member_indx = np.arange(start_mem, end_mem)
    gpcc_subensemble = xr.open_mfdataset(path_list,
                                         concat_dim='ens_mem',
                                         combine='nested')
    gpcc_subensemble = gpcc_subensemble.assign_coords({'ens_mem': member_indx})
    gpcc_subensemble.to_netcdf(data_dir + f'obsLE_chunk{j}.nc')

obsLE_total = xr.open_mfdataset(data_dir + 'obsLE_chunk*.nc')
obsLE_total.to_netcdf(data_dir + 'obsLE.nc')
