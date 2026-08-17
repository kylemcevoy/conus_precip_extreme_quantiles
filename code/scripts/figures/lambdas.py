import calendar
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

data_dir = '/home/data/projects/conus_precip_extremes/obsLE/gpcc/'
plot_dir = '/home/data/projects/conus_precip_extremes/plots/figures/'

lambdas = xr.open_dataarray(data_dir + 'optim_transform_params.nc')

months = np.arange(1, 12)
months = np.concatenate([[12], months])

month_name = [calendar.month_abbr[month] for month in months]

row = months // 3
row[0] = 0
col = months % 3

cmap = mpl.cm.viridis
levels = [0, 0.26, 0.34, 0.51, 0.76, 1.01]
tick_levels = levels[:-1] + np.diff(levels) / 2
norm = mpl.colors.BoundaryNorm(levels, cmap.N)

fig, ax = plt.subplots(figsize=(14, 8),
                       nrows=4, 
                       ncols=3, 
                       dpi=400,
                       constrained_layout=True,
                       subplot_kw={'projection': ccrs.PlateCarree()})

for i, month in enumerate(months):
    axis = ax[row[i], col[i]]
    (lambdas.sel(month=month)
     .plot(ax=axis,
           cmap=cmap,
           norm=norm,
           add_colorbar=False
                 ))
    axis.coastlines()
    axis.add_feature(cf.BORDERS)
    axis.set_title(month_name[i])
cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
             label=r'$\lambda$', shrink=0.8)
cbar.set_ticks(ticks=tick_levels, labels = ["1/4", "1/3", "1/2", "3/4", "1"])

fig.savefig(plot_dir + 'gpcc_lambdas.png')
