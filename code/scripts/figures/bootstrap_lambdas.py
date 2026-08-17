import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xarray as xr

proj_dir = '/home/data/projects/conus_precip_extremes/'
synthLE_dir = proj_dir + 'synthLE/cesm2/'
output_dir = synthLE_dir + 'analysis/q95/monthly/mm/'
plot_dir = proj_dir + 'plots/figures/'

path_list = [output_dir + f'synthLE_monthly_q95_mem{mem:02}.nc' 
             for mem in np.arange(50)]

synth_q95 = xr.open_mfdataset(path_list)
synth_q95 = synth_q95['precip']
synth_q95_dec = synth_q95.sel(month=12)

synth_q95_sd = synth_q95.groupby('cesm2_mem').std(ddof=1, dim='ens_mem')
synth_q95_sd_dec = synth_q95_sd.sel(month=12)

cesm2_LE = xr.open_dataarray(proj_dir + 'cesm2/cesm2_PRECT_processed.nc')
cesm2_LE_dec = cesm2_LE.sel(time=cesm2_LE['time.month'] == 12)

cesm2LE_q95_dec = cesm2_LE_dec.quantile(q=0.95, dim='time')
cesm2LE_q95_dec_sd = cesm2LE_q95_dec.std(ddof=1, dim='ens_mem')

frac_diff = (synth_q95_sd_dec - cesm2LE_q95_dec_sd) / cesm2LE_q95_dec_sd
frac_diff_df = frac_diff.to_dataframe(name='frac_diff')

lam_path = [synthLE_dir + f'mem{mem:02}/optim_transform_params.nc' 
            for mem in np.arange(50)]

lam_test = xr.open_mfdataset(lam_path,
                             combine='nested',
                             concat_dim='cesm2_mem')

lam_test = lam_test.compute()
lam_test = lam_test['lam']
lam_test = lam_test.assign_coords({'cesm2_mem': synth_q95_sd_dec.cesm2_mem.values})
lam_test

lam_dec = lam_test.sel(month=12)

lam_dec_df = lam_dec.to_dataframe()
synth_q95_sd_df = synth_q95_sd_dec.to_dataframe()

synth_q95_sd_df = synth_q95_sd_df.drop(labels='month', axis='columns')

lam_dec_df = lam_dec_df.drop(labels='month', axis='columns')

lam_dec_df_nona = lam_dec_df.dropna()
synth_q95_sd_nona = synth_q95_sd_df.dropna()
frac_diff_nona = frac_diff_df.dropna()

frac_diff_nona = frac_diff_df.drop('month', axis='columns')

joined_df = lam_dec_df_nona.join(frac_diff_nona)
joined_df = joined_df.drop('quantile', axis='columns')

mpl.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(12, 8), dpi=400)

sns.boxplot(joined_df, ax=ax, x='lam', y='frac_diff')
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(['1/4', '1/3', '1/2', '2/3', '3/4', '1'])
ax.set_ylabel('Fractional Difference in 0.95 Quantile Standard Deviation')
ax.set_xlabel(r'Box-Cox Transformation Parameter $\lambda$')

xmin, xmax = ax.get_xlim()
ax.hlines([0.1, -0.1], xmin=xmin, xmax=xmax, linestyles='--', color='red')

fig.savefig(plot_dir + 'lambda_bootstrap_study.png')

joined_df.groupby('lam').describe()

plt.close('all')

sns.histplot(lam_dec_df_nona['lam'])

gpcc_lam = xr.open_dataarray(proj_dir + 'obsLE/gpcc/optim_transform_params.nc')
gpcc_dec = gpcc_lam.sel(month=12)

lam_dec_df = lam_dec_df.dropna()
gpcc_dec_df = gpcc_dec.to_dataframe()
gpcc_dec_df = gpcc_dec_df.dropna()

fig, ax = plt.subplots()
sns.countplot(lam_dec_df, 
              ax=ax,
              x='lam',
              stat='proportion',
              label='CESM2')
sns.countplot(gpcc_dec_df, 
              ax=ax, 
              x='lam',
              stat='proportion', 
              alpha=0.5, 
              label='GPCC')

ax.set_xticklabels(['1/4', '1/3', '1/2', '2/3', '3/4', '1'])
ax.set_xlabel(r'$\lambda$')

fig.savefig(plot_dir + 'dec_lambda_prop_comp.png')

nan_mask = ~(lam_dec.isnull().any('cesm2_mem'))
lam_025_prop = (lam_dec == 0.25).mean('cesm2_mem').where(nan_mask)

mpl.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(nrows=1, 
                       ncols=2,
                       figsize=(16, 8),
                       dpi=400,
                       constrained_layout=True,
                       subplot_kw={'projection': ccrs.PlateCarree()})

gpcc_dec.plot(ax=ax[0],
              levels=[0, 0.26, 0.34, 0.51, 0.67, 0.76, 1.01],
              cbar_kwargs={'orientation': 'horizontal'})
ax[0].set_title('GPCC December Lambda')
ax[0].coastlines()
ax[0].add_feature(cf.BORDERS)

lam_025_prop.plot(ax=ax[1],
                  levels=np.linspace(0, 1 + 1e-6, 11),
                  cbar_kwargs={'orientation': 'horizontal'})
ax[1].set_title('CESM2 Member December Proportion Lambda = 0.25')
ax[1].coastlines()
ax[1].add_feature(cf.BORDERS)

fig.savefig(plot_dir + 'december_lambda_map_comp.png')

plt.close('all')