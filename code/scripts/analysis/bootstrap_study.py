import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

q95_dir = '/home/data/projects/conus_precip_extremes/synthLE/analysis/synth_q95/dec/'

mem00 = xr.open_dataarray(q95_dir + 'synthLE_mem00_dec_q95.nc')

mem00.var(dim='synth_ens_mem', ddof=1).plot()

np.arange(50)

q95_var_list = []
for mem in np.arange(50):
    single_member = xr.open_dataarray(q95_dir + f'synthLE_mem{mem:02}_dec_q95.nc')
    q95_var_list.append(single_member.var(dim='synth_ens_mem', ddof=1))

q95_var_da = xr.concat(q95_var_list, dim='ens_mem')

q95_synth_mean_var = q95_var_da.mean('ens_mem')

q95_synth_mean_var.plot()

cesm2_LE = xr.open_dataarray('/home/data/projects/conus_precip_extremes/cesm2/cesm2_total_precip_processed.nc')

cesm2_LE_dec = cesm2_LE.sel(time=cesm2_LE['time.month'] == 12)

cesm2LE_q95_dec = cesm2_LE_dec.quantile(q=0.95, dim='time')
cesm2LE_q95_dec_var = cesm2LE_q95_dec.var(dim='ens_mem', ddof=1)

fig, ax = plt.subplots(dpi=400,
                       subplot_kw={'projection': ccrs.PlateCarree()})
cesm2LE_q95_dec_var.plot(ax=ax,
                         extend='max',
                         levels=np.linspace(0, 2, 9),
                         cbar_kwargs={'shrink': 0.6,
                                      'label': r'variance of 0.95 quant. $[(mm/day)^2]$'})
ax.set_title('Variance of Dec. 0.95 quant. across CESM2-LE')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.STATES)
ax.coastlines()

fig, ax = plt.subplots(dpi=400,
                       subplot_kw={'projection': ccrs.PlateCarree()})
q95_synth_mean_var.plot(ax=ax,
                         extend='max',
                         levels=np.linspace(0, 2, 9),
                         cbar_kwargs={'shrink': 0.6,
                                      'label': r'variance of 0.95 quant. $[(mm/day)^2]$'})
ax.set_title('Mean Variance of Dec. 0.95 quant. across the 50 Synth-LE')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.STATES)
ax.coastlines()



var_ratio = q95_synth_mean_var / cesm2LE_q95_dec_var
log_var_ratio = np.log(var_ratio)

fig, ax = plt.subplots(dpi=400,
                       subplot_kw={'projection': ccrs.PlateCarree()})
(var_ratio).plot(ax=ax,
                 levels=np.linspace(0.5, 2, 16),
                 cbar_kwargs={'shrink': 0.5,
                              'label': 'variance ratio [ ]'})
ax.set_title('Var. Ratio of mean Synth-LE over CESM2-LE var. in 0.95 quantile')
plt.close()

fig, ax = plt.subplots(dpi=400,
                       subplot_kw={'projection': ccrs.PlateCarree()})
(log_var_ratio).plot(ax=ax,
                     extend='both',
                     levels=[np.log(1/2), np.log(2/3), np.log(5/6), np.log(1), np.log(1.2), np.log(3/2), np.log(2)],
                     cbar_kwargs={'shrink': 0.5,
                                  'label': 'log variance ratio [ ]'})
ax.set_title('Log Var. Ratio of Dec. 0.95 quant. (Synth-LE over CESM2-LE)')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.STATES)
ax.coastlines()

var_ratio_mem1 = q95_var_da.isel(ens_mem=0) / cesm2LE_q95_dec_var
log_var_ratio_mem1 = np.log(var_ratio_mem1)

fig, ax = plt.subplots(dpi=400,
                       subplot_kw={'projection': ccrs.PlateCarree()})
(log_var_ratio_mem1).plot(ax=ax,
                     extend='both',
                     levels=[np.log(1/2), np.log(2/3), np.log(5/6), np.log(1), np.log(1.2), np.log(3/2), np.log(2)],
                     cbar_kwargs={'shrink': 0.5,
                                  'label': 'log variance ratio [ ]'})
ax.set_title('Log Var. Ratio of Dec. 0.95 quant. (Synth-LE mem 1 over CESM2-LE)')
ax.add_feature(cf.BORDERS)
ax.coastlines()

var_ratio_mem14 = q95_var_da.isel(ens_mem=13) / cesm2LE_q95_dec_var
log_var_ratio_mem14 = np.log(var_ratio_mem14)



fig, ax = plt.subplots(dpi=400,
                       subplot_kw={'projection': ccrs.PlateCarree()})
(log_var_ratio_mem14).plot(ax=ax,
                     extend='both',
                     levels=[np.log(1/2), np.log(2/3), np.log(5/6), np.log(1), np.log(1.2), np.log(3/2), np.log(2)],
                     cbar_kwargs={'shrink': 0.5,
                                  'label': 'log variance ratio [ ]'})
ax.set_title('Log Var. Ratio of Dec. 0.95 quant. (Synth-LE mem 14 over CESM2-LE)')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.STATES)
ax.coastlines()

fig, ax = plt.subplots(dpi=400,
                       subplot_kw={'projection': ccrs.PlateCarree()})
q95_var_da.isel(ens_mem=13).plot(ax=ax,
                         levels=np.linspace(0, 2, 9),
                         cbar_kwargs={'shrink': 0.6,
                                      'label': r'variance of 0.95 quant. $[(mm/day)^2]$'})
ax.set_title('Variance of Dec. 0.95 quant. Synth-LE from CESM2 ens. mem. 14')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.STATES)
ax.coastlines()

fig, ax = plt.subplots(dpi=400,
                       subplot_kw={'projection': ccrs.PlateCarree()})
q95_var_da.isel(ens_mem=24).plot(ax=ax,
                         levels=np.linspace(0, 2, 9),
                         cbar_kwargs={'shrink': 0.6,
                                      'label': r'variance of 0.95 quant. $[(mm/day)^2]$'})
ax.set_title('Variance of Dec. 0.95 quant. Synth-LE from CESM2 ens. mem. 25')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.STATES)
ax.coastlines()