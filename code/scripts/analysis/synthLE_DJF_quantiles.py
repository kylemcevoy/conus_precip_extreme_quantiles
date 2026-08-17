# Script to calculate the 0.95 quantile of monthly precip. (for each month
# individually) for each member of the Synth-LE generated from each 
# CESM2-LE member.
import numpy as np
import xarray as xr

synthLE_dir = '/home/data/projects/conus_precip_extremes/synthLE/cesm2/'
output_dir_q95 = synthLE_dir + 'analysis/q95/djf/'
output_dir_q05 = synthLE_dir + 'analysis/q05/djf/'
# output_dir_q05 = 'analysis/q05/monthly/mm/'

# for mem in np.arange(50):
#     print(mem)
#     #Divide SynthLE into 10 chunks
#     path_lists = []
#     for j in range(10):
#         path_lists.append([synthLE_dir + f'mem{mem:02}/obsLE_member{i:04}.nc' 
#                        for i in np.arange(1 + (j * 100), 101 + (j * 100))])

#     for j, path_list in enumerate(path_lists):
#         start_mem = 1 + (j * 100)
#         end_mem = 101 + (j * 100)
#         member_indx = np.arange(start_mem, end_mem)
#         synthLE_subensemble = xr.open_mfdataset(path_list,
#                                             concat_dim='ens_mem',
#                                             combine='nested')
#         synthLE_subensemble = synthLE_subensemble.assign_coords(
#             {'ens_mem': member_indx}
#             )
        
#         days_in_month = synthLE_subensemble.time.dt.days_in_month
#         synthLE_monthly_totals = (synthLE_subensemble * days_in_month)
        
#         output_path = output_dir_q95 + f'synthLE_monthly_q95_mem{mem:02}_chunk{j + 1}.nc'
        
#         synthLE_chunk_q95 = (synthLE_monthly_totals.groupby('time.month')
#                              .quantile(0.95, dim='time'))
#         desc_str = (
#                 '0.95 quantile for each month of monthly precip [mm] data' 
#                 ' for synthetic ensemble members'
#                 f' generated from CESM2-LE ensemble members in chunk {j}. '
#                 'See analysis/synthLE_monthly_quantiles.py for script.'
#                 )
#         synthLE_chunk_q95 = synthLE_chunk_q95.assign_attrs(
#             {'description': desc_str,
#              'units': 'mm/day'}
#             )
        
#         synthLE_chunk_q95.to_netcdf(synthLE_dir + output_path)
    
#     chunk_dir = synthLE_dir + output_dir_q95
#     chunk_path = chunk_dir + f'synthLE_monthly_q95_mem{mem:02}_chunk*.nc'
#     synth_chunks = xr.open_mfdataset(chunk_path)
#     synth_chunks = synth_chunks.expand_dims(dim={'cesm2_mem': [mem]},
#                                             axis=0)
#     synth_chunks.to_netcdf(chunk_dir + f'synthLE_monthly_q95_mem{mem:02}.nc')
    
for mem in np.arange(50):
    print(mem)
    #Divide SynthLE into 10 chunks
    path_lists = []
    for j in range(10):
        path_lists.append([synthLE_dir + f'mem{mem:02}/obsLE_member{i:04}.nc' 
                       for i in np.arange(1 + (j * 100), 101 + (j * 100))])

    for j, path_list in enumerate(path_lists):
        start_mem = 1 + (j * 100)
        end_mem = 101 + (j * 100)
        member_indx = np.arange(start_mem, end_mem)
        synthLE_subensemble = xr.open_mfdataset(path_list,
                                            concat_dim='ens_mem',
                                            combine='nested')
        synthLE_subensemble = synthLE_subensemble.assign_coords(
            {'ens_mem': member_indx}
            )
        
        days_in_month = synthLE_subensemble.time.dt.days_in_month
        synthLE_monthly_totals = (synthLE_subensemble * days_in_month)
        synthLE_seasonal_totals = synthLE_monthly_totals.resample({'time': 'QS-DEC'}).sum('time')
        synthLE_DJF = synthLE_seasonal_totals.sel(time=synthLE_seasonal_totals['time.month'] == 12)
        synthLE_DJF = synthLE_DJF.sel(time=slice('1920-12-01', '2019-12-01'))
        synthLE_DJF_quantiles = synthLE_DJF.quantile(q=[0.05, 0.95], dim='time')
        
        output_path_q95 = output_dir_q95 + f'synthLE_monthly_q95_mem{mem:02}_chunk{j + 1}.nc'
        output_path_q05 = output_dir_q05 + f'synthLE_monthly_q05_mem{mem:02}_chunk{j + 1}.nc'
        
        synthLE_DJF_q95 = synthLE_DJF_quantiles.sel(quantile=0.95)
        synthLE_DJF_q05 = synthLE_DJF_quantiles.sel(quantile=0.05)
        
        desc_q95_str = (
                '0.95 quantile for each month of monthly precip [mm] data' 
                ' for synthetic ensemble members'
                f' generated from CESM2-LE ensemble members in chunk {j}. '
                'See analysis/synthLE_monthly_quantiles.py for script.'
                )
        desc_q05_str = (
                '0.05 quantile for each month of monthly precip [mm] data' 
                ' for synthetic ensemble members'
                f' generated from CESM2-LE ensemble members in chunk {j}. '
                'See analysis/synthLE_monthly_quantiles.py for script.'
        )
        synthLE_DJF_q95 = synthLE_DJF_q95.assign_attrs(
            {'description': desc_q95_str,
             'units': 'mm/day'}
            )
        
        synthLE_DJF_q95.to_netcdf(output_path_q95)
        
        synthLE_DJF_q05 = synthLE_DJF_q05.assign_attrs(
            {'description': desc_q05_str,
             'units': 'mm/day'}
            )
        
        synthLE_DJF_q05.to_netcdf(output_path_q05)
    
    chunk_dir_q95 = output_dir_q95
    chunk_path_q95 = chunk_dir_q95 + f'synthLE_monthly_q95_mem{mem:02}_chunk*.nc'
    synth_chunks = xr.open_mfdataset(chunk_path_q95)
    synth_chunks = synth_chunks.expand_dims(dim={'cesm2_mem': [mem]},
                                            axis=0)
    synth_chunks.to_netcdf(chunk_dir_q95 + f'synthLE_monthly_q95_mem{mem:02}.nc')
    
    chunk_dir_q05 = output_dir_q05
    chunk_path_q05 = chunk_dir_q05 + f'synthLE_monthly_q05_mem{mem:02}_chunk*.nc'
    synth_chunks = xr.open_mfdataset(chunk_path_q05)
    synth_chunks = synth_chunks.expand_dims(dim={'cesm2_mem': [mem]},
                                            axis=0)
    synth_chunks.to_netcdf(chunk_dir_q05 + f'synthLE_monthly_q05_mem{mem:02}.nc')
    
