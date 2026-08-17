import calendar
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

data_dir = '/home/data/projects/conus_precip_extremes/obsLE/gpcc_cvdp/'
plot_dir = '/home/data/projects/conus_precip_extremes/plots/figures/'

regress_out = xr.open_dataset(data_dir + 'regression_output.nc')

y_reconstruct = regress_out['residuals'] + regress_out['fitted_values']

y_var = y_reconstruct.groupby('time.month').var()
SS_total = y_var * 101

SS_error = regress_out['RSS']

R_squared = 1 - (SS_error / SS_total)

cmap_levels = np.linspace(0, 0.5, 11)
cmap = cm.viridis
norm = mpl.colors.BoundaryNorm(cmap_levels, cmap.N)

mpl.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(nrows=4, 
                       ncols=3,
                       dpi=400,
                       figsize=(10, 6),
                       constrained_layout=True,
                       subplot_kw={'projection': ccrs.PlateCarree()})
month_seq = np.hstack([[12], np.arange(1, 12)])
for i, axis in enumerate(ax.flatten()):
    month = month_seq[i]
    R_squared.sel(month=month).plot(ax=axis,
                                   cmap=cmap,
                                   norm=norm,
                                   add_colorbar=False)
    axis.coastlines()
    axis.add_feature(cf.BORDERS)
    axis.set_title(calendar.month_name[month])
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                    pad=0.01,
                    ax=ax, 
                    shrink=0.8)
cbar.set_label(label=r'$R^2$')

fig.savefig(plot_dir + 'model_R2_gpcc_cvdp.png')

plt.close("all")
