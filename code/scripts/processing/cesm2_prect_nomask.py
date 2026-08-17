# script to process each CESM2 precipitation components into a single large ensemble 
# netcdf covering the contiguous united states.

import pandas as pd
import numpy as np
import xarray as xr

cesm2_dir = '/home/data/CESM2/LE/monthly/'
output_dir = '/home/data/projects/conus_precip_extremes/cesm2/'

lat_min = 24.5
lat_max = 50
lon_min = 235
lon_max = 295

start_year = '1920'
end_year = '2020'

def extract_cesm2_data(var, 
                       data_dir, 
                       output_dir, 
                       start_year,
                       end_year,
                       lon_min,
                       lon_max,
                       lat_min,
                       lat_max
                       ):
    
    # Find the smoothed biomass burning ensemble members
    iso_years = [str(year) for year in 1011 + 20 * np.arange(10)]
    rep_years = ['1231', '1251', '1281', '1301']

    replicates = [str(rep).zfill(3) for rep in np.arange(11, 21)]

    iso_ens_mems = [data_dir + '*' + year + '*.nc' for year in iso_years]
    rep_ens_mems = [data_dir + '*' + year + '.' + rep + '*.nc' 
                    for year in rep_years
                    for rep in replicates]

    ens_mems = iso_ens_mems + rep_ens_mems

    cesm2_list = []
    for i, ens_mem in enumerate(ens_mems):
        print(i)
        cesm2_mem = xr.open_mfdataset(ens_mem)
        cesm2_mem = cesm2_mem[var].load()
        cesm2_list.append(cesm2_mem)

    cesm2_var_le = xr.concat(cesm2_list, dim='ens_mem')

    new_time = pd.date_range('1850-01-01', '2100-12-01', freq='MS')

    cesm2_var_le = cesm2_var_le.assign_coords({'time': new_time})

    cesm2_var_le = cesm2_var_le.sel(time=slice(start_year, end_year),
                                    lon = slice(lon_min, lon_max),
                                    lat = slice(lat_min, lat_max))

    cesm2_var_le = cesm2_var_le.assign_coords({'ens_mem': np.arange(50)})

    cesm2_var_le = (1000 * 60 * 60 * 24) * cesm2_var_le
    cesm2_var_le = cesm2_var_le.assign_attrs({'units': 'mm/day'})

    cesm2_var_le.to_netcdf(output_dir + f'cesm2_{var}_processed_nomask.nc')

extract_cesm2_data(var='PRECT',
                       data_dir=cesm2_dir + 'PRECT' + '/',
                       output_dir=output_dir,
                       start_year=start_year,
                       end_year=end_year,
                       lon_min=lon_min,
                       lon_max=lon_max,
                       lat_min=lat_min,
                       lat_max=lat_max)
