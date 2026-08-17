import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xarray as xr

proj_dir = '/home/data/projects/conus_precip_extremes/'
synthLE_dir = proj_dir + 'synthLE/cesm2/'
#synthLE_dir = proj_dir + 'synthLE/gamma/'

output_dir = synthLE_dir + 'analysis/q95/'
plot_dir = proj_dir + 'plots/figures/'

path_list = [output_dir + f'synthLE_monthly_q95_mem{mem:02}.nc' 
             for mem in np.arange(50)]

synth_q95 = xr.open_mfdataset(path_list)
synth_q95 = synth_q95['__xarray_dataarray_variable__']
synth_q95 = synth_q95.rename('precip')
synth_q95_dec = synth_q95.sel(month=12)

synth_q95_sd = synth_q95.groupby('cesm2_mem').std(ddof=1, dim='ens_mem')

synth_dec_means = synth_q95_dec.groupby('cesm2_mem').mean('ens_mem')
synth_dec_overall_mean = synth_dec_means.mean('cesm2_mem')

synth_q95_sd_dec = synth_q95_sd.sel(month=12).compute()
synth_q95_mean_sd = synth_q95_sd.mean('cesm2_mem')
synth_q95_mean_sd_dec = synth_q95_mean_sd.sel(month=12)
synth_q95_sd_sd_dec = synth_q95_sd.sel(month=12).std(ddof=1, dim='cesm2_mem')

local_sd = synth_q95_sd.sel(lat=34.5, lon=360-118.5, method='nearest')
local_sd_dec = local_sd.sel(month=12).compute()

local_mean_dec = synth_dec_means.sel(lat=34.5, lon=360-118.5, method='nearest')
local_mean_dec = local_mean_dec.compute()

local_lon2 = 273.75
local_lat2 = synth_q95_sd_dec.lat.values[11]
local_sd2 = synth_q95_sd_dec.sel(lat=local_lat2, lon=local_lon2)

cesm2_LE = xr.open_dataarray(proj_dir + 'cesm2/cesm2_PRECT_processed.nc')
cesm2_LE_dec = cesm2_LE.sel(time=cesm2_LE['time.month'] == 12)
# convert from mm/day to mm/month
cesm2_LE_dec = cesm2_LE_dec * 31

cesm2LE_q95_dec = cesm2_LE_dec.quantile(q=0.95, dim='time')

cesm2_q95_mean = cesm2LE_q95_dec.mean('ens_mem')

cesm2LE_q95_dec_sd = cesm2LE_q95_dec.std(ddof=1, dim='ens_mem')

local_cesm2_mean_dec = cesm2_q95_mean.sel(lat=34.5,
                                          lon=360-118.5,
                                          method='nearest')
local_cesm2_sd_dec = cesm2LE_q95_dec_sd.sel(lat=34.5,
                                            lon=360-118.5,
                                            method='nearest')

local_cesm2_sd_dec2 = cesm2LE_q95_dec_sd.sel(lat=local_lat2,
                                            lon=local_lon2)

local_lat = local_cesm2_sd_dec.lat.values
height = (cesm2_LE.lat.values[10] - cesm2_LE.lat.values[9])
bottom = local_lat - height / 2

local_lon = local_cesm2_sd_dec.lon.values
width = (cesm2_LE.lon.values[5] - cesm2_LE.lon.values[4])
left = local_lon - width / 2

height2 = (cesm2_LE.lat.values[11] - cesm2_LE.lat.values[10])
bottom2 = local_lat - height2 / 2
width2 = (cesm2_LE.lon.values[2] - cesm2_LE.lon.values[1])
left2 = local_lon2 - width2 / 2

frac_change_in_mean = ((synth_dec_overall_mean - cesm2_q95_mean) 
                     / cesm2_q95_mean)

mean_mean_change = frac_change_in_mean.mean().compute().values

frac_change_in_sd = ((synth_q95_mean_sd_dec - cesm2LE_q95_dec_sd) 
                     / cesm2LE_q95_dec_sd)

mean_sd_change = frac_change_in_sd.mean().compute().values

cmap_levels = np.linspace(-0.4, 0.4, 17)
cmap = cm.RdBu_r
norm = mpl.colors.BoundaryNorm(cmap_levels, cmap.N, extend='both')

sd_cmap = cm.viridis
sd_norm = mpl.colors.BoundaryNorm(np.linspace(0, 12, 13), sd_cmap.N)

rectangle = mpatches.Rectangle((left, bottom),
                                 width=width,
                                 height=height,
                                 fill=False,
                                 edgecolor='black',
                                 transform=ccrs.PlateCarree())

rectangle2 = mpatches.Rectangle((left, bottom),
                                 width=width,
                                 height=height,
                                 fill=False,
                                 edgecolor='black',
                                 transform=ccrs.PlateCarree())

rectangle3 = mpatches.Rectangle((left2, bottom2),
                                 width=width2,
                                 height=height2,
                                 fill=False,
                                 edgecolor='tab:green',
                                 transform=ccrs.PlateCarree())

rectangle4 = mpatches.Rectangle((left2, bottom2),
                                 width=width2,
                                 height=height2,
                                 fill=False,
                                 edgecolor='tab:green',
                                 transform=ccrs.PlateCarree())

mpl.rcParams.update({'font.size': 11})
fig = plt.figure(figsize=(16, 12), constrained_layout=True, dpi=400)
gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[2, 1])
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
ax3 = fig.add_subplot(gs[0, 1])
ax4 = fig.add_subplot(gs[1, 1])




(frac_change_in_sd).plot(ax=ax1,
                 cmap=cmap,
                 norm=norm,
                 add_colorbar=False)
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=ax1,
                    pad=0.01,
                    shrink=0.8,
                    orientation='horizontal')
cbar.set_label('fractional change of mean standard deviation [ ]',
               fontsize=14)
ax1.add_patch(rectangle)
ax1.add_patch(rectangle3)
ax1.coastlines()
ax1.set_title('')
ax1.add_feature(cf.BORDERS)
ax1.annotate(f'mean: {mean_sd_change:.3f}', [285, 30], fontsize=14)
ax1.set_title('a)', loc='left', fontsize=18)

(synth_q95_sd_sd_dec).plot(ax=ax2,
                 cmap=sd_cmap,
                 norm=sd_norm,
                 add_colorbar=False)
cbar2 = fig.colorbar(cm.ScalarMappable(norm=sd_norm, cmap=sd_cmap),
                    ax=ax2,
                    pad=0.01,
                    shrink=0.8,
                    orientation='horizontal')
cbar2.set_label('standard deviation of quantile standard deviations [mm]',
               fontsize=14)
ax2.add_patch(rectangle2)
ax2.add_patch(rectangle4)
ax2.coastlines()
ax2.set_title('')
ax2.add_feature(cf.BORDERS)
ax2.set_title('b)', loc='left', fontsize=18)

sns.histplot(local_sd_dec, ax=ax3, binrange=[0, 36], binwidth=3, color='tab:purple')
ax3.axvline(x=local_cesm2_sd_dec.compute().values, 
            color='red', 
            linestyle='--',
            label='CESM2-LE s.d.')
ax3.axvline(x=local_sd_dec.mean().values, 
            color='black', 
            linestyle='--',
            label='synth-LE mean s.d.'
            )
ax3.legend()
ax3.set_ylim((0.0, 15.75))
ax3.set_xlabel('standard deviations of the 0.95 quantile [mm]',
               fontsize=14)
ax3.set_title('c)', loc='left', fontsize=18)
ax3.tick_params(axis='both', labelsize=12)

sns.histplot(local_sd2, ax=ax4, binrange=[0, 36], binwidth=2, color='tab:cyan')
ax4.axvline(x=local_cesm2_sd_dec2.compute().values, 
            color='red', 
            linestyle='--',
            label='CESM2-LE s.d.')
ax4.axvline(x=local_sd2.mean().values, 
            color='black', 
            linestyle='--',
            label='synth-LE mean s.d.'
            )
ax4.legend()
ax4.set_xlabel('standard deviations of the 0.95 quantile [mm]',
               fontsize=14)
ax4.set_title('d)', loc='left', fontsize=18)
ax4.tick_params(axis='both', labelsize=12)

fig.savefig(plot_dir + 'sd_decq95_synthLE_cesm2LE_comp.png')


### Just panel c)

mpl.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(dpi=400, figsize=(12, 8))

sns.histplot(local_sd_dec, ax=ax, binrange=[0, 36], binwidth=3, color='tab:purple')
ax.axvline(x=local_cesm2_sd_dec.compute().values, 
            color='red', 
            linestyle='--',
            label='CESM2-LE s.d.')
ax.axvline(x=local_sd_dec.mean().values, 
            color='black', 
            linestyle='--',
            label='synth-LE mean s.d.'
            )
ax.legend()
ax.set_ylim((0.0, 15.75))
ax.set_xlabel('standard deviations of the 0.95 quantile [mm]')
ax.tick_params(axis='both', labelsize=12)


### Just panel d)

fig, ax = plt.subplots(dpi=400, figsize=(12, 8))

sns.histplot(local_sd2, ax=ax, binrange=[0, 36], binwidth=2, color='tab:cyan')
ax.axvline(x=local_cesm2_sd_dec2.compute().values, 
            color='red', 
            linestyle='--',
            label='CESM2-LE s.d.')
ax.axvline(x=local_sd2.mean().values, 
            color='black', 
            linestyle='--',
            label='synth-LE mean s.d.'
            )
ax.legend()
ax.set_xlabel('standard deviations of the 0.95 quantile [mm]')


### mean bias

fig = plt.figure(figsize=(16, 6), constrained_layout=True)
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])
ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(gs[1])

cmap_levels = np.linspace(-0.05, 0.05, 5)
cmap = cm.RdBu_r
norm = mpl.colors.BoundaryNorm(cmap_levels, cmap.N, extend='both')

(frac_change_in_mean).plot(ax=ax1,
                 cmap=cmap,
                 norm=norm,
                 add_colorbar=False)
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=ax1,
                    pad=0.01,
                    shrink=0.8,
                    orientation='horizontal')
cbar.set_label('fractional change of mean q95 [ ]',
               fontsize=14)
ax1.set_title('')
ax1.coastlines()
ax1.add_feature(cf.BORDERS)
ax1.annotate(f'mean: {mean_mean_change:.3f}', [285, 30], fontsize=14)
ax1.set_title('a)', loc='left', fontsize=18)

sns.histplot(local_mean_dec, ax=ax2, binrange=[3.5, 6], binwidth=0.25)
ax2.axvline(x=local_cesm2_mean_dec.compute().values, 
            color='red', 
            linestyle='--',
            label='CESM2-LE mean q95')
ax2.axvline(x=local_mean_dec.mean().values, 
            color='black', 
            linestyle='--',
            label='synth-LE mean q95'
            )
ax2.legend()
ax2.set_xlabel('mean values of the 0.95 quantile',
               fontsize=14)
ax2.set_title('b)', loc='left', fontsize=18)
ax2.tick_params(axis='both', labelsize=12)


fig.savefig(plot_dir + 'mean_bias_decq95_synthLE_cesm2LE_comp.png')


# synth_q95_variance = synth_q95.groupby('cesm2_mem').var(ddof=1, dim='ens_mem')
# synth_q95_mean_var = synth_q95_variance.mean('cesm2_mem')
# synth_q95_mean_var_dec = synth_q95_mean_var.sel(month=12)

# cesm2LE_q95_dec_var = cesm2LE_q95_dec.var(dim='ens_mem', ddof=1)

# cbar_label = r'variance of 0.95 quant. $[(\mathrm{mm/day})^2]$'

# fig, ax = plt.subplots(dpi=400,
#                        subplot_kw={'projection': ccrs.PlateCarree()})
# cesm2LE_q95_dec_var.plot(ax=ax,
#                          extend='max',
#                          levels=np.linspace(0, 2, 9),
#                          cbar_kwargs={'shrink': 0.6,
#                                       'label': cbar_label})
# ax.set_title('Variance of Dec. 0.95 quant. across CESM2-LE')
# ax.add_feature(cf.BORDERS)
# ax.coastlines()

# fig, ax = plt.subplots(dpi=400,
#                        subplot_kw={'projection': ccrs.PlateCarree()})
# synth_q95_mean_var_dec.plot(ax=ax,
#                          extend='max',
#                          levels=np.linspace(0, 2, 9),
#                          cbar_kwargs={'shrink': 0.6,
#                                       'label': cbar_label})
# ax.set_title('Mean Variance of Dec. 0.95 quant. across the 50 Synth-LE')
# ax.add_feature(cf.BORDERS)
# ax.coastlines()



# var_ratio = synth_q95_mean_var_dec / cesm2LE_q95_dec_var
# per_change_in_var = (synth_q95_mean_var_dec - cesm2LE_q95_dec_var) / cesm2LE_q95_dec_var
# log_var_ratio = np.log(var_ratio)
# mean_var_change = per_change_in_var.mean().compute().values

# fig, ax = plt.subplots(dpi=400,
#                        subplot_kw={'projection': ccrs.PlateCarree()})
# (var_ratio).plot(ax=ax,
#                  levels=np.linspace(0.5, 2, 7),
#                  cbar_kwargs={'shrink': 0.5,
#                               'label': 'variance ratio [ ]'})
# ax.set_title('Var. Ratio of mean Synth-LE over CESM2-LE var. in 0.95 quantile')
# plt.close('all')

# fig, ax = plt.subplots(dpi=400,
#                        subplot_kw={'projection': ccrs.PlateCarree()})
# (per_change_in_var).plot(ax=ax,
#                  levels=np.linspace(-0.8, 0.8, 9),
#                  extend='both',
#                  cbar_kwargs={'shrink': 0.8,
#                               'pad': 0.01,
#                               'orientation': 'horizontal',
#                               'label': 'fractional change [ ]'})
# ax.set_title('')
# ax.coastlines()
# ax.add_feature(cf.BORDERS)
# ax.annotate(f'mean: {mean_change:.3f}', [285, 30])
# fig.savefig(plot_dir + 'fractional_change_decq95.png')

# plt.close('all')

# cmap_levels = [np.log(1/2),
#                np.log(2/3),
#                np.log(5/6),
#                np.log(1),
#                np.log(1.2),
#                np.log(3/2),
#                np.log(2)]
# cmap = cm.RdBu_r
# norm = mpl.colors.BoundaryNorm(cmap_levels, cmap.N, extend='both')

# fig, ax = plt.subplots(dpi=400,
#                        constrained_layout=True,
#                        subplot_kw={'projection': ccrs.PlateCarree()})
# (log_var_ratio).plot(ax=ax,
#                      extend='both',
#                      cmap=cmap,
#                      norm=norm,
#                      add_colorbar=False)
# cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
#                     ax=ax,
#                     orientation='horizontal')
# cbar.set_label('log ratio of variances [ ]')
# cbar.set_ticklabels([r'$\log(1/2)$',
#                      r'$\log(2/3)$',
#                      r'$\log(5/6)$',
#                      r'$\log(1)$',
#                      r'$\log(6/5)$',
#                      r'$\log(3/2)$',
#                      r'$\log(2)$'])
# ax.set_title('Log Var. Ratio of Dec. 0.95 quant. (Synth-LE mem 1 over CESM2-LE)')
# ax.add_feature(cf.BORDERS)
# ax.coastlines()

plt.close('all')


fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True, dpi=400,
                       subplot_kw={'projection': ccrs.PlateCarree()})


(frac_change_in_sd).plot(ax=ax,
                 cmap=cmap,
                 norm=norm,
                 add_colorbar=False)
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=ax,
                    pad=0.01,
                    shrink=0.8,
                    orientation='horizontal')
cbar.set_label('fractional change of mean standard deviation [ ]',
               fontsize=14)
ax.coastlines()
ax.set_title('')
ax.add_feature(cf.BORDERS)
ax.annotate(f'mean: {mean_sd_change:.3f}', [285, 30], fontsize=14)
ax.set_title('a)', loc='left', fontsize=18)


fig, ax3 = plt.subplots(dpi=400, figsize=(12, 8))

sns.histplot(local_sd_dec, ax=ax3, binrange=[0, 36], binwidth=3, color='tab:purple')
ax3.axvline(x=local_cesm2_sd_dec.compute().values, 
            color='red', 
            linestyle='--',
            label='CESM2-LE s.d.')
ax3.axvline(x=local_sd_dec.mean().values, 
            color='black', 
            linestyle='--',
            label='synth-LE mean s.d.'
            )
ax3.legend()
ax3.set_ylim((0.0, 15.75))
ax3.set_xlabel('standard deviations of the 0.95 quantile [mm]',
               fontsize=14)
ax3.set_title('c)', loc='left', fontsize=18)
ax3.tick_params(axis='both', labelsize=12)

fig, ax4 = plt.subplots(figsize=(12, 8), dpi=400)

sns.histplot(local_sd2, ax=ax4, binrange=[0, 36], binwidth=2, color='tab:cyan')
ax4.axvline(x=local_cesm2_sd_dec2.compute().values, 
            color='red', 
            linestyle='--',
            label='CESM2-LE s.d.')
ax4.axvline(x=local_sd2.mean().values, 
            color='black', 
            linestyle='--',
            label='synth-LE mean s.d.'
            )
ax4.legend()
ax4.set_xlabel('standard deviations of the 0.95 quantile [mm]',
               fontsize=14)
ax4.set_title('d)', loc='left', fontsize=18)
ax4.tick_params(axis='both', labelsize=12)
