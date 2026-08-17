import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib as mpl
#import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

from scipy.stats import ecdf

proj_dir = '/home/data/projects/conus_precip_extremes/'
data_dir = proj_dir + 'obsLE/gpcc_cvdp/'
plot_dir = proj_dir + 'plots/figures/'

gpcc = xr.open_dataarray(proj_dir + 'gpcc/gpcc_totals.nc')

ortho_modes_df = xr.open_dataset(data_dir + 'ortho_modes.nc')
ortho_modes_df = ortho_modes_df.to_pandas()

surr_modes = xr.open_dataset(data_dir + 'surrogate_modes.nc')

obsLE = xr.open_dataarray(data_dir + 'obsLE_totals.nc')
# laish = obsLE.sel(lat=34.5, lon=-118.5)
# laish.to_netcdf(data_dir + 'laish.nc')

# del obsLE

laish = xr.open_dataarray(data_dir + 'laish.nc')

laish_gpcc = (gpcc.sel(lat=34.5, lon=-118.5)
              .sel(time=gpcc['time.month'] == 12))
mode_dec = ortho_modes_df.loc[ortho_modes_df.index.month == 12]
laish_gpcc_djf = laish_gpcc.sel(time=laish_gpcc['time.month'].isin([12, 1, 2]))

laish_djf = laish.sel(time=laish['time.month'].isin([12, 1, 2]))

surr_modes_djf = surr_modes.sel(
    time=surr_modes['time.month'].isin([12, 1, 2])
    )

enso_pos_pna_neg = (surr_modes_djf['enso'] > 1) & (surr_modes_djf['pna'] < -1)
enso_neg_pna_pos = (surr_modes_djf['enso'] < -1) & (surr_modes_djf['pna'] > 1)
enso_pna_neutral = (
    (surr_modes_djf['enso'] > -1) & (surr_modes_djf['enso'] < 1) 
    & (surr_modes_djf['pna'] > -1) & (surr_modes_djf['pna'] < 1)
    ) 

laish_djf_large = laish_djf.where(enso_pos_pna_neg)
laish_djf_large = laish_djf_large.values.flatten()
laish_djf_neutral = laish_djf.where(enso_pna_neutral)
laish_djf_neutral = laish_djf_neutral.values.flatten()
laish_djf_small = laish_djf.where(enso_neg_pna_pos)
laish_djf_small = laish_djf_small.values.flatten()

laish_large_q95 = np.nanquantile(laish_djf_large, q=0.95)
laish_neutral_q95 = np.nanquantile(laish_djf_neutral, q=0.95)
laish_small_q95 = np.nanquantile(laish_djf_small, q=0.95)

laish_djf_large = laish_djf_large[~np.isnan(laish_djf_large)]
laish_djf_neutral = laish_djf_neutral[~np.isnan(laish_djf_neutral)]
laish_djf_small = laish_djf_small[~np.isnan(laish_djf_small)]

obsLE_djf = obsLE.sel(time=obsLE['time.month'].isin([12, 1, 2]))

obsLE_djf_large = obsLE_djf.where(enso_pos_pna_neg).quantile(0.95, 
                                                             dim=['ens_mem', 'time'])
obsLE_djf_neutral = obsLE_djf.where(enso_pna_neutral).quantile(0.95, 
                                                               dim=['ens_mem', 'time'])

obsLE_djf_q95_diff = obsLE_djf_large - obsLE_djf_neutral

betas = xr.open_dataset(data_dir + 'beta.nc')
params = xr.open_dataset(data_dir + 'optim_transform_params.nc')

cmap_big = mpl.cm.BrBG
levels_big = np.linspace(-80, 80, 17)
norm_big = mpl.colors.BoundaryNorm(levels_big, cmap_big.N, extend='both')

laish_lat = laish.lat.values
laish_lon = laish.lon.values

laish_betas = betas.sel(lat=laish_lat, lon=laish_lon)
laish_params = params.sel(lat=laish_lat, lon=laish_lon)['lam']

mode_list = list(surr_modes.data_vars.keys())

bins_comp = np.linspace(-0.125, 15.125, 15 * 4 + 2)
laish_djf_flat = laish_djf.values.flatten()

colors = sns.color_palette('tab10')
bins = np.linspace(-12.5, 412.5, 19)
palette = sns.color_palette(None, n_colors=3, as_cmap=True)

mpl.rcParams.update({'font.size': 22})
fig = plt.figure(layout='constrained', figsize=(20, 10))
subfigs = fig.subfigures(1, 2)

ax1 = subfigs[0].subplots(3, 1, sharex=True)
ax2 = subfigs[1].subplots(2, 1, 
                          height_ratios=[2,1],
                          subplot_kw=dict(projection=ccrs.PlateCarree()),)

sns.histplot(laish_djf_small, 
             ax=ax1[0], 
             label='ENSO-/PNA+',
             stat='density', 
             bins=bins, 
             color=palette[2])
sns.histplot(laish_djf_neutral,
             label='neutral ENSO/neutral PNA',
             ax=ax1[1],
             stat='density', 
             bins=bins, 
             color=palette[1])
sns.histplot(laish_djf_large,
             label='ENSO+/PNA-',
             ax=ax1[2], 
             stat='density', 
             bins=bins, 
             color=palette[0])

ax1[0].set_ylim([0, 0.013])
ax1[0].set_title('ENSO-/PNA+', fontsize=22)
ax1[0].set_ylabel('Density', fontsize=22)
ax1[0].vlines(laish_small_q95, 
              ymin=0, 
              ymax=0.013, 
              color=palette[2],
              linestyle='--')

ax1[1].set_ylim([0, 0.013])
ax1[1].set_title('Neutral ENSO/Neutral PNA', fontsize=22)
ax1[1].set_ylabel('Density', fontsize=22)
ax1[1].vlines(laish_neutral_q95, 
              ymin=0, 
              ymax=0.013,
              color=palette[1],
              linestyle='--')

ax1[2].set_ylim([0, 0.013])
ax1[2].set_title('ENSO+/PNA-', fontsize=22)
ax1[2].set_xlabel('precipitation [mm]', fontsize=22)
ax1[2].set_ylabel('Density', fontsize=22)
ax1[2].vlines(laish_large_q95, 
              ymin=0, 
              ymax=0.013,
              color=palette[0],
              linestyle='--')
ax1[0].set_title('a)', loc='left', fontsize=22)

obsLE_djf_q95_diff.plot(ax=ax2[0],
                      cmap=cmap_big,
                      norm=norm_big,
                      add_colorbar=False)

ax2[0].coastlines()
ax2[0].add_feature(cf.BORDERS)
ax2[0].add_feature(cf.STATES)

cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap_big, norm=norm_big),
                    pad=0.05, 
                    ax=ax2[0],
                    shrink=0.8,
                    orientation='horizontal')
cbar.set_label('DJF precipitation composite [mm]', fontsize=22)
ax2[0].set_title('')
ax2[0].set_title('b)', loc='left', fontsize=22)
ax2[0].coastlines()
ax2[0].add_feature(cf.BORDERS)
ax2[1].set_axis_off()

fig.savefig(plot_dir + 'extremes_shift2.png')
