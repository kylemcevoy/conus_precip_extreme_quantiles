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

output_dir = synthLE_dir + 'analysis/q05/djf/'
plot_dir = proj_dir + 'plots/figures/'

path_list = [output_dir + f'synthLE_monthly_q05_mem{mem:02}.nc' 
             for mem in np.arange(50)]

cesm2_LE = xr.open_dataarray(proj_dir + 'cesm2/cesm2_PRECT_processed.nc')
nan_mask = ~(cesm2_LE.isnull().any(['time', 'ens_mem']))
cesm2_LE_totals = cesm2_LE * cesm2_LE.time.dt.days_in_month
cesm2_LE_seasonal = cesm2_LE_totals.resample({'time': 'QS-DEC'}).sum("time")
cesm2_LE_seasonal = cesm2_LE_seasonal.where(nan_mask)
cesm2_LE_djf = cesm2_LE_seasonal.sel(time=cesm2_LE_seasonal['time.month'] == 12)
cesm2_LE_djf = cesm2_LE_djf.sel(time=slice('1920', '2019'))

synth_q05 = xr.open_mfdataset(path_list)
synth_q05 = synth_q05['precip']

synth_q05_sd = synth_q05.groupby('cesm2_mem').std(ddof=1, dim='ens_mem')
synth_q05_sd = synth_q05_sd.where(nan_mask)

synth_means = synth_q05.groupby('cesm2_mem').mean('ens_mem')
synth_overall_mean = synth_means.mean('cesm2_mem')

synth_q05_mean_sd = synth_q05_sd.mean('cesm2_mem')
synth_q05_sd_sd_djf = synth_q05_sd.std(ddof=1, dim='cesm2_mem')
synth_q05_sd_sd_djf = synth_q05_sd_sd_djf.where(nan_mask)

local_sd_djf = synth_q05_sd.sel(lat=34.5, lon=360-118.5, method='nearest').compute()

local_mean_djf = synth_means.sel(lat=34.5, lon=360-118.5, method='nearest')
local_mean_djf = local_mean_djf.compute()

local_lon2 = 278.75
local_lat2 = synth_q05_sd.lat.values[4]
local_sd2_djf = synth_q05_sd.sel(lat=local_lat2, lon=local_lon2)

cesm2LE_q05_djf = cesm2_LE_djf.quantile(q=0.05, dim='time')

cesm2_q05_mean = cesm2LE_q05_djf.mean('ens_mem')

cesm2LE_q05_djf_sd = cesm2LE_q05_djf.std(ddof=1, dim='ens_mem')

local_cesm2_mean_djf = cesm2_q05_mean.sel(lat=34.5,
                                          lon=360-118.5,
                                          method='nearest')
local_cesm2_sd_djf = cesm2LE_q05_djf_sd.sel(lat=34.5,
                                            lon=360-118.5,
                                            method='nearest')

local_cesm2_sd_djf2 = cesm2LE_q05_djf_sd.sel(lat=local_lat2,
                                            lon=local_lon2)

local_lat = local_cesm2_sd_djf.lat.values
height = (cesm2_LE.lat.values[10] - cesm2_LE.lat.values[9])
bottom = local_lat - height / 2

local_lon = local_cesm2_sd_djf.lon.values
width = (cesm2_LE.lon.values[5] - cesm2_LE.lon.values[4])
left = local_lon - width / 2

height2 = (cesm2_LE.lat.values[5] - cesm2_LE.lat.values[4])
bottom2 = local_lat2 - height2 / 2
width2 = (cesm2_LE.lon.values[2] - cesm2_LE.lon.values[1])
left2 = local_lon2 - width2 / 2

frac_change_in_mean = ((synth_overall_mean - cesm2_q05_mean) 
                     / cesm2_q05_mean)

mean_mean_change = frac_change_in_mean.mean().compute().values

frac_change_in_sd = ((synth_q05_mean_sd - cesm2LE_q05_djf_sd) 
                     / cesm2LE_q05_djf_sd)

mean_sd_change = frac_change_in_sd.mean().compute().values

cmap_levels = np.linspace(-0.4, 0.4, 17)
cmap = cm.RdBu_r
norm = mpl.colors.BoundaryNorm(cmap_levels, cmap.N, extend='both')

sd_cmap = cm.viridis
sd_norm = mpl.colors.BoundaryNorm(np.linspace(0, 4, 9), sd_cmap.N)

rectangle = mpatches.Rectangle((left, bottom),
                                 width=width,
                                 height=height,
                                 fill=False,
                                 edgecolor='tab:red',
                                 linewidth=3,
                                 transform=ccrs.PlateCarree())

rectangle2 = mpatches.Rectangle((left, bottom),
                                 width=width,
                                 height=height,
                                 fill=False,
                                 edgecolor='tab:red',
                                 linewidth=3,
                                 transform=ccrs.PlateCarree())

rectangle3 = mpatches.Rectangle((left2, bottom2),
                                 width=width2,
                                 height=height2,
                                 fill=False,
                                 edgecolor='tab:red',
                                 linewidth=3,
                                 transform=ccrs.PlateCarree())

rectangle4 = mpatches.Rectangle((left2, bottom2),
                                 width=width2,
                                 height=height2,
                                 fill=False,
                                 edgecolor='tab:red',
                                 linewidth=3,
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
ax1.coastlines()
ax1.set_title('')
ax1.add_feature(cf.BORDERS)
ax1.annotate(f'mean: {mean_sd_change:.3f}', [285, 30], fontsize=14)
ax1.set_title('a)', loc='left', fontsize=18)

(synth_q05_sd_sd_djf).plot(ax=ax2,
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

sns.histplot(local_sd_djf, ax=ax3, color='tab:purple')
ax3.axvline(x=local_cesm2_sd_djf.compute().values, 
            color='red', 
            linestyle='--',
            label='CESM2-LE s.d.')
ax3.axvline(x=local_sd_djf.mean().values, 
            color='black', 
            linestyle='--',
            label='synth-LE mean s.d.'
            )
ax3.legend()
ax3.set_ylim((0.0, 15.75))
ax3.set_xlabel('standard deviations of the 0.05 quantile [mm]',
               fontsize=14)
ax3.set_title('c)', loc='left', fontsize=18)
ax3.tick_params(axis='both', labelsize=12)

sns.histplot(local_sd2_djf, ax=ax4, color='tab:cyan')
ax4.axvline(x=local_cesm2_sd_djf2.compute().values, 
            color='red', 
            linestyle='--',
            label='CESM2-LE s.d.')
ax4.axvline(x=local_sd2_djf.mean().values, 
            color='black', 
            linestyle='--',
            label='synth-LE mean s.d.'
            )
ax4.legend()
ax4.set_xlabel('standard deviations of the 0.05 quantile [mm]',
               fontsize=14)
ax4.set_title('d)', loc='left', fontsize=18)
ax4.tick_params(axis='both', labelsize=12)

fig.savefig(plot_dir + 'sd_q05_djf_synthLE_cesm2LE_comp.png')


