import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

gpcc_dir = '/home/data/projects/conus_precip_extremes/gpcc/'
data_dir = '/home/data/projects/conus_precip_extremes/obsLE/gpcc_cvdp/'
#output_dir = data_dir + 'analysis/'
plot_dir = '/home/data/projects/conus_precip_extremes/plots/figures/'

gpcc = xr.open_dataarray(gpcc_dir + 'gpcc_mmday.nc')

obsLE = xr.open_mfdataset(data_dir + 'obsLE_chunk*.nc')
#obsLE.to_netcdf(data_dir + 'obsLE.nc')
#obsLE = xr.open_dataarray(data_dir + 'obsLE.nc')

# obsLE_total_quantile = (obsLE.groupby('time.month')
#                         .quantile(0.95, dim=['time', 'ens_mem']))

# obsLE_total_quantile.to_netcdf(output_dir + 'obsLE_total_month_q95.nc')

ortho_modes_df = xr.open_dataset(data_dir + 'ortho_modes.nc')
ortho_modes_df = ortho_modes_df.to_pandas()

surr_modes = xr.open_dataset(data_dir + 'surrogate_modes.nc')

surr_djf = surr_modes.sel(time=surr_modes['time.month'].isin([12, 1, 2]))

laish = obsLE.sel(lat=34.5, lon=-118.5)
laish = laish['precip']

laish_djf = laish.sel(time=laish['time.month'].isin([12, 1, 2]))

enso_pos_pna_neg = (surr_modes['enso'] > 1) & (surr_modes['pna'] < -1)
enso_neg_pna_pos = (surr_modes['enso'] < -1) & (surr_modes['pna'] > 1)
enso_pna_neutral = ((surr_modes['enso'] > -1) & (surr_modes['enso'] < 1) & 
                    (surr_modes['pna'] > -1) & (surr_modes['pna'] < 1)) 
laish_djf_large = laish_djf.where(enso_pos_pna_neg).values.flatten()
laish_djf_neutral = laish_djf.where(enso_pna_neutral).values.flatten()
laish_djf_small = laish_djf.where(enso_neg_pna_pos).values.flatten()

laish_djf_mean = laish_djf.mean().values

laish_djf_large = laish_djf_large[~np.isnan(laish_djf_large)]
laish_djf_neutral = laish_djf_neutral[~np.isnan(laish_djf_neutral)]
laish_djf_small = laish_djf_small[~np.isnan(laish_djf_small)]

bins = np.linspace(-0.5, 12.5, 13)
palette = sns.color_palette(None, n_colors=3, as_cmap=True)

fig, ax = plt.subplots(figsize=(12, 10),
                       nrows=3,
                       ncols=1,
                       dpi=400,
                       sharex=True)

sns.histplot(laish_djf_small, 
             ax=ax[0], 
             label='ENSO-/PNA+',
             stat='density', 
             bins=bins, 
             color=palette[2])
sns.histplot(laish_djf_neutral,
             label='neutral ENSO/neutral PNA',
             ax=ax[1],
             stat='density', 
             bins=bins, 
             color=palette[1])
sns.histplot(laish_djf_large,
             label='ENSO+/PNA-',
             ax=ax[2], 
             stat='density', 
             bins=bins, 
             color=palette[0])

ax[0].set_ylim([0, 0.4])
ax[0].set_title('ENSO-/PNA+', fontsize=14)
ax[0].set_ylabel('Density', fontsize=14)

ax[1].set_ylim([0, 0.4])
ax[1].set_title('Neutral ENSO/Neutral PNA', fontsize=14)
ax[1].set_ylabel('Density', fontsize=14)

ax[2].set_ylim([0, 0.4])
ax[2].set_title('ENSO+/PNA-', fontsize=14)
ax[2].set_xlabel('precipitation [mm/day]', fontsize=14)
ax[2].set_ylabel('Density', fontsize=14)

fig.savefig(plot_dir + 'mode_histograms')
