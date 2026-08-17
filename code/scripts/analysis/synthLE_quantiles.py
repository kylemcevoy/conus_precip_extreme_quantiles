# Script to calculate the 0.95 quantile of monthly precip. (across all months)
# for each member of the Synth-LE generated from each CESM2-LE member.

import numpy as np
import xarray as xr

synthLE_dir = ('/home/data/projects/conus_precip_extremes/synthLE/' 
               'cesm2/full_ensemble/')

for mem in np.arange(50):
    print(mem)
    mem_path = f'mem{mem:02}/obsLE_member*.nc'
    output_path = f'analysis/q95/all_months/synthLE_mem{mem:02}_q95.nc'
    synthLE_mem = xr.open_mfdataset(synthLE_dir + mem_path,
                                    combine='nested',
                                    concat_dim='synth_ens_mem')

    synthLE_mem_q95 = synthLE_mem['precip'].quantile(0.95, dim='time')
    
    desc_str = ('0.95 quantile of precip [mm/day] for synthetic ensemble members'
                f' generated from CESM2-LE ensemble member {mem:02}. '
                'See analysis/synthLE_quantiles.py for script.')
    
    synthLE_mem_q95 = synthLE_mem_q95.assign_attrs({'description': desc_str,
                                                    'units': 'mm/day'})
    
    synthLE_mem_q95.to_netcdf(synthLE_dir + output_path)
