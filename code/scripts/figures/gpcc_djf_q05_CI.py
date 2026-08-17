import calendar
import cartopy.crs as ccrs
import cartopy.feature as cf
import cmasher
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

mpl.rcParams.update({'font.size': 20})

data_dir = '/home/data/projects/conus_precip_extremes/obsLE/gpcc_cvdp/'
output_dir = data_dir + 'analysis/'
plot_dir = '/home/data/projects/conus_precip_extremes/plots/figures/'

gpcc_mem = xr.open_dataarray(data_dir + 'obsLE_member0001.nc')
nan_mask = ~(gpcc_mem.isnull().any('time'))

gpcc_q05 = xr.open_dataarray(output_dir + 'gpcc_djf_q05.nc')
obsLE_q05 = xr.open_dataarray(output_dir + 'obsLE_djf_q05.nc')

gpcc_q05 = gpcc_q05.where(nan_mask)
obsLE_q05 = obsLE_q05.where(nan_mask)

obsLE_lb = obsLE_q05.quantile(0.025, dim='ens_mem').where(nan_mask)
obsLE_ub = obsLE_q05.quantile(0.975, dim='ens_mem').where(nan_mask)
ci_spread = obsLE_ub - obsLE_lb 

upper_spread = obsLE_ub - gpcc_q05
lower_spread = obsLE_lb - gpcc_q05

cmap_levels = np.linspace(0, 400, 17)
cmap = cm.viridis
norm = mpl.colors.BoundaryNorm(cmap_levels, cmap.N, extend='max')

cmap_spread = cm.plasma
norm_spread = mpl.colors.BoundaryNorm(np.linspace(0, 100, 11),
                                      cmap_spread.N,
                                      extend='max')

pos_diff = cmasher.get_sub_cmap(cm.BrBG, 0.5, 1)
neg_diff = cmasher.get_sub_cmap(cm.BrBG, 0, 0.5)

norm_pos = mpl.colors.BoundaryNorm(np.linspace(0, 100, 11),
                                    pos_diff.N,
                                    extend='max')
norm_neg = mpl.colors.BoundaryNorm(np.linspace(-50, 0, 6),
                                    neg_diff.N,
                                    extend='min')

font_size = 24

fig, ax = plt.subplots(nrows=2, 
                       ncols=3,
                       dpi=400,
                       figsize=(18, 8.5), 
                       constrained_layout=True,
                       subplot_kw={'projection':ccrs.PlateCarree()})


obsLE_lb.plot(ax=ax[0, 0],
                            cmap=cmap,
                            norm=norm,
                            add_colorbar=False)

clb = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
             ax=ax[0, 0],
             orientation='horizontal',)
clb.set_label(label='precipitation [mm]', fontsize=font_size)

ax[0, 0].set_title('95% CI Lower Bound',
                   fontsize=font_size)
ax[0, 0].coastlines()
ax[0, 0].add_feature(cf.BORDERS)

gpcc_q05.plot(ax=ax[0, 1], 
                            cmap=cmap,
                            norm=norm,
                            add_colorbar=False)

clb = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
             ax=ax[0, 1],
             orientation='horizontal',)
clb.set_label(label='precipitation [mm]', fontsize=font_size)

ax[0, 1].set_title('GPCC Observed DJF 0.05 Quantile', fontsize=font_size)
ax[0, 1].coastlines()
ax[0, 1].add_feature(cf.BORDERS)

obsLE_ub.plot(ax=ax[0, 2],
                            cmap=cmap,
                            norm=norm,
                            add_colorbar=False)

clb = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
             ax=ax[0, 2],
             orientation='horizontal',)
clb.set_label(label='precipitation [mm]', fontsize=font_size)

ax[0, 2].set_title('95% CI Upper Bound', fontsize=font_size)
ax[0, 2].coastlines()
ax[0, 2].add_feature(cf.BORDERS)

ci_spread.plot(ax=ax[1, 1],
                             cmap=cmap_spread,
                             norm=norm_spread,
                             add_colorbar=False)

clb = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap_spread, norm=norm_spread),
                   ax=ax[1, 1],
                   orientation='horizontal')
clb.set_label(label='precipitation spread [mm]', fontsize=font_size)

ax[1, 1].set_title('Confidence Interval Spread', fontsize=font_size)
ax[1, 1].coastlines()
ax[1, 1].add_feature(cf.BORDERS)

lower_spread.plot(ax=ax[1, 0],
                                cmap=neg_diff,
                                norm=norm_neg,
                                add_colorbar=False)

spread_clb = fig.colorbar(mpl.cm.ScalarMappable(cmap=neg_diff, 
                                                norm=norm_neg),
             ax=ax[1, 0],
             orientation='horizontal',)
spread_clb.set_label(label='precipitation difference [mm]', fontsize=font_size)

ax[1, 0].set_title('Lower Bound minus GPCC quantile',
                   fontsize=font_size)
ax[1, 0].coastlines()
ax[1, 0].add_feature(cf.BORDERS)

upper_spread.plot(ax=ax[1, 2],
                                cmap=pos_diff,
                                norm=norm_pos,
                                add_colorbar=False)

uspread_clb = fig.colorbar(mpl.cm.ScalarMappable(cmap=pos_diff, 
                                                norm=norm_pos),
             ax=ax[1, 2],
             orientation='horizontal',)
uspread_clb.set_label(label='precipitation difference [mm]', fontsize=font_size)

ax[1, 2].set_title('Upper Bound minus GPCC quantile',
                   fontsize=font_size)
ax[1, 2].coastlines()
ax[1, 2].add_feature(cf.BORDERS)

fig.savefig(plot_dir + 'gpcc_obsLE_djf_q05_CI.png')

plt.close("all")

cmap_increase = cm.viridis
norm_increase = mpl.colors.BoundaryNorm(np.linspace(0, 1, 11),
                                      cmap_increase.N)

ub_increase = ((obsLE_ub - gpcc_q05) / gpcc_q05)

fig, ax = plt.subplots(nrows=4, 
                       ncols=3,
                       dpi=400,
                       figsize=(10, 6),
                       constrained_layout=True,
                       subplot_kw={'projection': ccrs.PlateCarree()})

month_seq = np.hstack([[12], np.arange(1, 12)])
for i, axis in enumerate(ax.flatten()):
    month = month_seq[i]
    ub_increase.sel(month=month).plot(ax=axis,
                                   cmap=cmap_increase,
                                   norm=norm_increase,
                                   add_colorbar=False)
    axis.coastlines()
    axis.add_feature(cf.BORDERS)
    axis.set_title(calendar.month_name[month], fontsize=12)
cbar = fig.colorbar(cm.ScalarMappable(norm=norm_increase, cmap=cmap_increase),
                    pad=0.01,
                    ax=ax, 
                    shrink=0.8)
cbar.set_label(label='% Increase', fontsize=12)

fig.savefig(plot_dir + 'CI_ub_over_obs.png')