# Script to calculate the 0.95 quantile of monthly precip. (for each month
# individually) for each member of the Synth-LE generated from each 
# CESM2-LE member.
import numpy as np
import xarray as xr

synthLE_dir = ('/home/data/projects/conus_precip_extremes/synthLE/' 
               'gamma/')
output_dir_q95 = synthLE_dir + 'analysis/q95/'
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
                       for i in np.arange((j * 100), 100 + (j * 100))])

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
        
        #days_in_month = synthLE_subensemble.time.dt.days_in_month
        #synthLE_monthly_totals = (synthLE_subensemble * days_in_month)
        
        output_path = output_dir_q95 + f'synthLE_monthly_q95_mem{mem:02}_chunk{j + 1}.nc'
        
        synthLE_chunk_q95 = (synthLE_subensemble.groupby('time.month')
                             .quantile(0.95, dim='time'))
        desc_str = (
                '0.95 quantile for each month of monthly precip [mm] data' 
                ' for synthetic ensemble members'
                f' generated from CESM2-LE ensemble members in chunk {j}. '
                'See analysis/synthLE_monthly_quantiles.py for script.'
                )
        synthLE_chunk_q95 = synthLE_chunk_q95.assign_attrs(
            {'description': desc_str,
             'units': 'mm/day'}
            )
        
        synthLE_chunk_q95.to_netcdf(output_path)
    
    chunk_dir = output_dir_q95
    chunk_path = chunk_dir + f'synthLE_monthly_q95_mem{mem:02}_chunk*.nc'
    synth_chunks = xr.open_mfdataset(chunk_path)
    synth_chunks = synth_chunks.expand_dims(dim={'cesm2_mem': [mem]},
                                            axis=0)
    synth_chunks.to_netcdf(chunk_dir + f'synthLE_monthly_q95_mem{mem:02}.nc')