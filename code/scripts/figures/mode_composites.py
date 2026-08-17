import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pandas import Index

mpl.rcParams.update({'font.size': 16})

proj_dir = '/home/data/projects/conus_precip_extremes/'
gpcc_dir = proj_dir + 'gpcc/'
data_dir = proj_dir + 'obsLE/gpcc_cvdp/'
analysis_dir = data_dir + 'analysis/'
plot_dir = proj_dir + 'plots/figures/'

ortho_modes = xr.open_dataset(data_dir + 'ortho_modes.nc')
gpcc = xr.open_dataarray(gpcc_dir + 'gpcc_totals.nc')

surr_modes = xr.open_dataset(data_dir + 'surrogate_modes.nc')

obsLE = xr.open_dataarray(data_dir + 'obsLE_totals.nc')

###### Seasonal Average at the end #######
season_list = ['DJF', 'MAM', 'JJA', 'SON']

var_names = list(ortho_modes.data_vars)
var_names.pop(0)

nan_mask = ~gpcc.isnull().any('time')

mode_composite_list = []
for mode_name in var_names:
    mode = ortho_modes[mode_name]
    
    gpcc_pos = gpcc.sel(time=(mode > 1))
    gpcc_neg = gpcc.sel(time=(mode < -1))
    
    pos_avg = ((gpcc_pos)
                  .groupby('time.season')
                  .mean('time')
                  .where(nan_mask))
    neg_avg = ((gpcc_neg)
                  .groupby('time.season')
                  .mean('time')
                  .where(nan_mask))
    mode_composite_list.append(pos_avg - neg_avg)

season_list = season_list[0:2]
obsLE_composite_list = []
for mode_name in var_names:
    season_index = Index(season_list, name='season')
    season_composite_list = []
    surr_mode = surr_modes[mode_name]
    for season in season_list:
        surr_mode_season = surr_mode.sel(time=surr_mode['time.season'] == season)
        obsLE_season = obsLE.sel(time=obsLE['time.season'] == season)
        obsLE_pos = obsLE_season.where(surr_mode_season > 1)
        obsLE_neg = obsLE_season.where(surr_mode_season < -1)
        
        obsLE_pos_mean = obsLE_pos.mean(['time', 'ens_mem'])
        obsLE_neg_mean = obsLE_neg.mean(['time', 'ens_mem'])
        
        obsLE_composite_season = obsLE_pos_mean - obsLE_neg_mean
        season_composite_list.append(obsLE_composite_season)
    obsLE_mode_composite = xr.concat(season_composite_list, dim=season_index)
    obsLE_composite_list.append(obsLE_mode_composite)
    
cmap = mpl.cm.BrBG
levels = np.linspace(-60, 60, 13)
norm = mpl.colors.BoundaryNorm(levels, cmap.N, extend='both')

col_labels = ['GPCC', 'GPCC-Synth-LE']
row_labels = var_names


#### DJF ####

fig, ax = plt.subplots(dpi=400,
                       nrows = 4,
                       ncols = 2,
                       constrained_layout=True,
                       subplot_kw={'projection': ccrs.PlateCarree()})

for j in range(4):
    mode = var_names[j]
    mode_composite_list[j].sel(season='DJF').plot(ax = ax[j, 0],
                                                   cmap=cmap,
                                                   norm=norm,
                                                   add_colorbar=False)
    obsLE_composite_list[j].sel(season='DJF').plot(ax = ax[j, 1],
                                                   cmap=cmap,
                                                   norm=norm,
                                                   add_colorbar=False)
    ax[j, 0].set_title('')
    ax[j, 0].add_feature(cf.BORDERS)
    ax[j, 0].coastlines()
    ax[j, 0].set_yticks([])
    ax[j, 0].set_ylabel(mode.upper(),
                        rotation='horizontal',
                        ha='right',
                        fontsize=14)
    ax[j, 1].set_title('')
    ax[j, 1].add_feature(cf.BORDERS)
    ax[j, 1].coastlines()
    
for axis, col_label in zip(ax[0], col_labels):
    axis.set_title(col_label, fontsize=14)

cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, 
                                   norm=norm), 
             ax=ax,
             shrink=0.8)
cbar.set_label('precipitation composite [mm]',
                fontsize=14)
fig.savefig(plot_dir + 'composite_comp_djf_end.png')

#### MAM ######

fig, ax = plt.subplots(dpi=400,
                       nrows = 4,
                       ncols = 2,
                       constrained_layout=True,
                       subplot_kw={'projection': ccrs.PlateCarree()})

for j in range(4):
    mode = var_names[j]
    mode_composite_list[j].sel(season='MAM').plot(ax = ax[j, 0],
                                                   cmap=cmap,
                                                   norm=norm,
                                                   add_colorbar=False)
    obsLE_composite_list[j].sel(season='MAM').plot(ax = ax[j, 1],
                                                   cmap=cmap,
                                                   norm=norm,
                                                   add_colorbar=False)
    ax[j, 0].set_title('')
    ax[j, 0].add_feature(cf.BORDERS)
    ax[j, 0].coastlines()
    ax[j, 0].set_yticks([])
    ax[j, 0].set_ylabel(mode.upper(),
                        rotation='horizontal',
                        ha='right',
                        fontsize=12)
    ax[j, 1].set_title('')
    ax[j, 1].add_feature(cf.BORDERS)
    ax[j, 1].coastlines()
    
for axis, col_label in zip(ax[0], col_labels):
    axis.set_title(col_label)

fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, 
                                   norm=norm), 
             ax=ax,
             shrink=0.8,
             label='precip. composite [mm]')
fig.savefig(plot_dir + 'composite_comp_mam_end.png')

#### JJA #####

fig, ax = plt.subplots(dpi=400,
                       nrows = 4,
                       ncols = 2,
                       constrained_layout=True,
                       subplot_kw={'projection': ccrs.PlateCarree()})

for j in range(4):
    mode = var_names[j]
    mode_composite_list[j].sel(season='JJA').plot(ax = ax[j, 0],
                                                   cmap=cmap,
                                                   norm=norm,
                                                   add_colorbar=False)
    obsLE_composite_list[j].sel(season='JJA').plot(ax = ax[j, 1],
                                                   cmap=cmap,
                                                   norm=norm,
                                                   add_colorbar=False)
    ax[j, 0].set_title('')
    ax[j, 0].add_feature(cf.BORDERS)
    ax[j, 0].coastlines()
    ax[j, 0].set_yticks([])
    ax[j, 0].set_ylabel(mode.upper(),
                        rotation='horizontal',
                        ha='right',
                        fontsize=12)
    ax[j, 1].set_title('')
    ax[j, 1].add_feature(cf.BORDERS)
    ax[j, 1].coastlines()
    
for axis, col_label in zip(ax[0], col_labels):
    axis.set_title(col_label)

fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, 
                                   norm=norm), 
             ax=ax,
             shrink=0.8,
             label='precip. composite [mm]')
fig.savefig(plot_dir + 'composite_comp_jja_end.png')

#### SON #####

fig, ax = plt.subplots(dpi=400,
                       nrows = 4,
                       ncols = 2,
                       constrained_layout=True,
                       subplot_kw={'projection': ccrs.PlateCarree()})

for j in range(4):
    mode = var_names[j]
    mode_composite_list[j].sel(season='SON').plot(ax = ax[j, 0],
                                                   cmap=cmap,
                                                   norm=norm,
                                                   add_colorbar=False)
    obsLE_composite_list[j].sel(season='SON').plot(ax = ax[j, 1],
                                                   cmap=cmap,
                                                   norm=norm,
                                                   add_colorbar=False)
    ax[j, 0].set_title('')
    ax[j, 0].add_feature(cf.BORDERS)
    ax[j, 0].coastlines()
    ax[j, 0].set_yticks([])
    ax[j, 0].set_ylabel(mode.upper(),
                        rotation='horizontal',
                        ha='right',
                        fontsize=12)
    ax[j, 1].set_title('')
    ax[j, 1].add_feature(cf.BORDERS)
    ax[j, 1].coastlines()
    
for axis, col_label in zip(ax[0], col_labels):
    axis.set_title(col_label)

fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, 
                                   norm=norm), 
             ax=ax,
             shrink=0.8,
             label='precip. composite [mm]')
fig.savefig(plot_dir + 'composite_comp_son_end.png')


#### DJF multi-mode composite

surr_modes_djf = surr_modes.sel(time=surr_modes['time.season'] == 'DJF')
obsLE_djf = obsLE.sel(time=obsLE['time.season'] == 'DJF')

west_coast_pos_conditions = ((surr_modes_djf['enso'] > 1) 
                         & (surr_modes_djf['pna'] < -1) 
                         & (surr_modes['nao'] < -1))

west_coast_neg_conditions = ((surr_modes_djf['enso'] < -1) 
                         & (surr_modes_djf['pna'] > 1) 
                         & (surr_modes['nao'] > 1))

west_coast_pos_conditions.sum().values
west_coast_neg_conditions.sum().values

obsLE_wc_pos = obsLE_djf.where(west_coast_pos_conditions).mean(['time', 'ens_mem'])
obsLE_wc_neg = obsLE_djf.where(west_coast_neg_conditions).mean(['time', 'ens_mem'])

obsLE_comp = obsLE_wc_pos - obsLE_wc_neg

cmap_big = mpl.cm.BrBG
levels_big = np.linspace(-80, 80, 17)
norm_big = mpl.colors.BoundaryNorm(levels_big, cmap_big.N, extend='both')

fig, ax = plt.subplots(dpi=400,
                       constrained_layout=True,
                       subplot_kw={'projection': ccrs.PlateCarree()})

obsLE_comp.plot(ax = ax,
                cmap=cmap_big,
                norm=norm_big,
                add_colorbar=False)
ax.set_title('')
ax.add_feature(cf.BORDERS)
ax.coastlines()

fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap_big, 
                                   norm=norm_big),
             pad=0.05, 
             ax=ax,
             shrink=0.8,
             orientation='horizontal',
             label='precipitation composite [mm]')
fig.savefig(plot_dir + 'west_coast_wet_composite.png')

florida_wet = ((surr_modes_djf['enso'] > 1) 
                & (surr_modes_djf['pdo'] > 1)
                & (surr_modes_djf['pna'] > 1) 
                & (surr_modes_djf['nao'] < -1))

florida_dry = ((surr_modes_djf['enso'] < -1) 
                & (surr_modes_djf['pdo'] < -1)
                & (surr_modes_djf['pna'] < -1) 
                & (surr_modes_djf['nao'] > 1))

florida_wet_comp = (obsLE_djf.where(florida_wet).mean(['time', 'ens_mem']) 
 - obsLE_djf.where(florida_dry).mean(['time', 'ens_mem']))

fig, ax = plt.subplots(dpi=400,
                       constrained_layout=True,
                       subplot_kw={'projection': ccrs.PlateCarree()})

florida_wet_comp.plot(ax = ax,
                cmap=cmap_big,
                norm=norm_big,
                add_colorbar=False)
ax.set_title('')
ax.add_feature(cf.BORDERS)
ax.coastlines()

fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap_big, 
                                   norm=norm_big),
             pad=0.05, 
             ax=ax,
             shrink=0.8,
             orientation='horizontal',
             label='precipitation composite [mm/day]')

fig.savefig(plot_dir + 'florida_wet_comp.png')


mpl.rcParams.update({'font.size': 20})
fig, (ax1, ax2) = plt.subplots(nrows=1,
                               ncols=2,
                               dpi=400,
                               figsize=(12, 4.2), 
                               constrained_layout=True,
                               subplot_kw={'projection': ccrs.PlateCarree()})

obsLE_comp.plot(ax = ax1,
                cmap=cmap_big,
                norm=norm_big,
                add_colorbar=False)

ax1.set_title('a) California Maximizing Composite',
              loc='left',
              fontsize=20)
ax1.coastlines()
ax1.add_feature(cf.STATES)
ax1.add_feature(cf.BORDERS)

florida_wet_comp.plot(ax=ax2,
                      cmap=cmap_big,
                      norm=norm_big,
                      add_colorbar=False)

ax2.set_title('b) Florida Maximizing Composite',
              loc='left',
              fontsize=20)
ax2.coastlines()
ax2.add_feature(cf.BORDERS)
ax2.add_feature(cf.STATES)

cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap_big, norm=norm_big),
                    pad=0.05, 
                    ax=(ax1, ax2),
                    shrink=0.6,
                    orientation='horizontal')
cbar.set_label('precipitation composite [mm]', fontsize=20)

fig.savefig(plot_dir + 'multimode_composites_states2.png')

ortho_modes_djf = ortho_modes.sel(time=ortho_modes['time.season'] == 'DJF')

((ortho_modes_djf['enso'] > 1) &
 (ortho_modes_djf['pna'] < - 1) &
 (ortho_modes_djf['nao'] < -1)).sum()

((ortho_modes_djf['enso'] < -1) &
 (ortho_modes_djf['pna'] > 1) &
 (ortho_modes_djf['nao'] > 1)).sum()

(gpcc.sel(time=gpcc['time.season'] == 'DJF').where(((ortho_modes_djf['enso'] > 1) &
 (ortho_modes_djf['pna'] < - 1) &
 (ortho_modes_djf['nao'] < -1))).mean('time') - gpcc.sel(time=gpcc['time.season'] == 'DJF').where(((ortho_modes_djf['enso'] < -1) &
 (ortho_modes_djf['pna'] > 1) &
 (ortho_modes_djf['nao'] > 1))).mean('time')).plot()

###### Seasonal Averaging First #######

## Helper Functions

# def convert_to_seasonal(data_array, mean=False):
#     days_in_month = data_array.time.dt.days_in_month
#     nan_mask = ~(data_array.isnull().any('time'))

#     season_resample = ((data_array * days_in_month)
#                     .resample(time='QS-DEC'))
#     season_agg = season_resample.sum('time')
        
#     if mean:
#         total_days = days_in_month.resample(time='QS-DEC').sum('time')
#         season_agg = season_agg / total_days
        
#     season_agg = season_agg.where(nan_mask)
#     season_agg = season_agg.sel(time=slice('1920-03-01', '2020-09-01'))
#     return season_agg

####### Divide combine gpcc_obsLE into 10 chunks on disk #########

# gpcc_obsLE = xr.open_mfdataset(data_dir + 'obsLE_member*.nc',
#                                concat_dim = 'ens_mem',
#                                combine='nested'
#                                )
# path_lists = []
# for j in range(10):
#     path_lists.append([data_dir + f'obsLE_member{i:04}.nc' 
#                        for i in np.arange(1 + (j * 100), 101 + (j * 100))])

# for j, path_list in enumerate(path_lists):
#     print(j)
#     start_mem = 1 + (j * 100)
#     end_mem = 101 + (j * 100)
#     member_indx = np.arange(start_mem, end_mem)
#     gpcc_subensemble = xr.open_mfdataset(path_list,
#                                          concat_dim='ens_mem',
#                                          combine='nested')
#     gpcc_subensemble = gpcc_subensemble.assign_coords({'ens_mem': member_indx})
#     gpcc_subensemble.to_netcdf(data_dir + f'obsLE_chunk{j}.nc')

# gpcc_obsLE = xr.open_mfdataset(data_dir + 'obsLE_chunk*.nc')
# gpcc_obsLE = gpcc_obsLE['var']

# gpcc_obsLE_season_mmday = convert_to_seasonal(gpcc_obsLE, mean=True)
# gpcc_obsLE_season_mmday.to_netcdf(analysis_dir + 'figures/gpcc_obsLE_seasonal_mmday_avg.nc')

# obsLE_season_mmday = xr.open_dataarray(analysis_dir + 'figures/gpcc_obsLE_seasonal_mmday_avg.nc')

# gpcc_season_mmday = convert_to_seasonal(gpcc, mean=True)

# seas_mode_list = []
# for mode_name in var_names:
#     mode = ortho_modes[mode_name]
#     mode_seas = (mode.resample(time='QS-DEC')
#                  .mean('time')
#                  .sel(time=slice('1920-03-01','2020-09-01')))
#     seas_mode_list.append(mode_seas)

# composite_list = []
# for mode in seas_mode_list:
#     gpcc_pos_mode = gpcc_season_mmday.sel(time=(mode > 1))
#     gpcc_neg_mode = gpcc_season_mmday.sel(time=(mode < -1))
#     gpcc_composite = (gpcc_pos_mode.groupby('time.season').mean('time') 
#                       - gpcc_neg_mode.groupby('time.season').mean('time'))
#     composite_list.append(gpcc_composite)
    

# surr_modes = surr_modes.rename({'ens_member': 'ens_mem'})
# surr_modes = surr_modes.assign_coords({'ens_mem': np.arange(1, 1001)})

# surr_seas_mode_list = []
# for mode_name in var_names:
#     mode = surr_modes[mode_name]
#     mode_seas = (mode.resample(time='QS-DEC')
#                  .mean('time')
#                  .sel(time=slice('1920-03-01','2020-09-01')))
#     surr_seas_mode_list.append(mode_seas)

# obsLE_composite_list = []
# for surr_mode in surr_seas_mode_list:
#     obsLE_pos = (obsLE_season_mmday
#                  .where(surr_mode > 1)
#                  .groupby('time.season')
#                  .mean('time')
#                  .mean('ens_mem'))
#     obsLE_neg = (obsLE_season_mmday
#                  .where(surr_mode < - 1)
#                  .groupby('time.season')
#                  .mean('time')
#                  .mean('ens_mem'))
#     obsLE_composite = obsLE_pos - obsLE_neg
#     obsLE_composite_list.append(obsLE_composite)

# cmap = mpl.cm.BrBG
# levels = np.linspace(-2, 2, 17)
# norm = mpl.colors.BoundaryNorm(levels, cmap.N, extend='both')

# col_labels = ['GPCC', 'Obs-LE']
# row_labels = var_names

# fig, ax = plt.subplots(dpi=400,
#                        nrows = 4,
#                        ncols = 2,
#                        constrained_layout=True,
#                        subplot_kw={'projection': ccrs.PlateCarree()})

# for j in range(4):
#     mode = var_names[j]
#     composite_list[j].sel(season='DJF').plot(ax = ax[j, 0],
#                                                    cmap=cmap,
#                                                    norm=norm,
#                                                    add_colorbar=False)
#     obsLE_composite_list[j].sel(season='DJF').plot(ax = ax[j, 1],
#                                                    cmap=cmap,
#                                                    norm=norm,
#                                                    add_colorbar=False)
#     ax[j, 0].set_title('')
#     ax[j, 0].add_feature(cf.BORDERS)
#     ax[j, 0].coastlines()
#     ax[j, 1].set_title('')
#     ax[j, 1].add_feature(cf.BORDERS)
#     ax[j, 1].coastlines()
    
# for axis, col_label in zip(ax[0], col_labels):
#     axis.set_title(col_label)
    
# fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
#              label='precip anom. [mm/day]')

# fig.text(x=-0.06, y=0.825, s=row_labels[0].upper(), fontsize='large')
# fig.text(x=-0.05, y=0.6, s=row_labels[1].upper(), fontsize='large')
# fig.text(x=-0.05, y=0.35, s=row_labels[2].upper(), fontsize='large')
# fig.text(x=-0.05, y=0.125, s=row_labels[3].upper(), fontsize='large')

# fig.savefig(plot_dir + 'composite_comp_djf.png')


# fig, ax = plt.subplots(dpi=400,
#                        nrows = 4,
#                        ncols = 2,
#                        constrained_layout=True,
#                        subplot_kw={'projection': ccrs.PlateCarree()})

# for j in range(4):
#     mode = var_names[j]
#     composite_list[j].sel(season='MAM').plot(ax = ax[j, 0],
#                                                    cmap=cmap,
#                                                    norm=norm,
#                                                    add_colorbar=False)
#     obsLE_composite_list[j].sel(season='MAM').plot(ax = ax[j, 1],
#                                                    cmap=cmap,
#                                                    norm=norm,
#                                                    add_colorbar=False)
#     ax[j, 0].set_title('')
#     ax[j, 0].add_feature(cf.BORDERS)
#     ax[j, 0].coastlines()
#     ax[j, 1].set_title('')
#     ax[j, 1].add_feature(cf.BORDERS)
#     ax[j, 1].coastlines()
    
# for axis, col_label in zip(ax[0], col_labels):
#     axis.set_title(col_label)
    
# fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
#              label='precip anom. [mm/day]')

# fig.text(x=-0.06, y=0.825, s=row_labels[0].upper(), fontsize='large')
# fig.text(x=-0.05, y=0.6, s=row_labels[1].upper(), fontsize='large')
# fig.text(x=-0.05, y=0.35, s=row_labels[2].upper(), fontsize='large')
# fig.text(x=-0.05, y=0.125, s=row_labels[3].upper(), fontsize='large')

# fig.savefig(plot_dir + 'composite_comp_mam.png')

# ##### JJA

# fig, ax = plt.subplots(dpi=400,
#                        nrows = 4,
#                        ncols = 2,
#                        constrained_layout=True,
#                        subplot_kw={'projection': ccrs.PlateCarree()})

# for j in [0, 1, 3]:
#     mode = var_names[j]
#     composite_list[j].sel(season='JJA').plot(ax = ax[j, 0],
#                                                    cmap=cmap,
#                                                    norm=norm,
#                                                    add_colorbar=False)
#     obsLE_composite_list[j].sel(season='JJA').plot(ax = ax[j, 1],
#                                                    cmap=cmap,
#                                                    norm=norm,
#                                                    add_colorbar=False)
#     ax[j, 0].set_title('')
#     ax[j, 0].add_feature(cf.BORDERS)
#     ax[j, 0].coastlines()
#     ax[j, 1].set_title('')
#     ax[j, 1].add_feature(cf.BORDERS)
#     ax[j, 1].coastlines()
    
# for axis, col_label in zip(ax[0], col_labels):
#     axis.set_title(col_label)
    
# fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
#              label='precip anom. [mm/day]')

# fig.text(x=-0.06, y=0.825, s=row_labels[0].upper(), fontsize='large')
# fig.text(x=-0.05, y=0.6, s=row_labels[1].upper(), fontsize='large')
# fig.text(x=-0.05, y=0.35, s=row_labels[2].upper(), fontsize='large')
# fig.text(x=-0.05, y=0.125, s=row_labels[3].upper(), fontsize='large')

# fig.savefig(plot_dir + 'composite_comp_jja.png')

# #### SON

# fig, ax = plt.subplots(dpi=400,
#                        nrows = 4,
#                        ncols = 2,
#                        constrained_layout=True,
#                        subplot_kw={'projection': ccrs.PlateCarree()})

# for j in range(4):
#     mode = var_names[j]
#     composite_list[j].sel(season='SON').plot(ax = ax[j, 0],
#                                                    cmap=cmap,
#                                                    norm=norm,
#                                                    add_colorbar=False)
#     obsLE_composite_list[j].sel(season='SON').plot(ax = ax[j, 1],
#                                                    cmap=cmap,
#                                                    norm=norm,
#                                                    add_colorbar=False)
#     ax[j, 0].set_title('')
#     ax[j, 0].add_feature(cf.BORDERS)
#     ax[j, 0].coastlines()
#     ax[j, 1].set_title('')
#     ax[j, 1].add_feature(cf.BORDERS)
#     ax[j, 1].coastlines()
    
# for axis, col_label in zip(ax[0], col_labels):
#     axis.set_title(col_label)
    
# fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
#              label='precip anom. [mm/day]')

# fig.text(x=-0.06, y=0.825, s=row_labels[0].upper(), fontsize='large')
# fig.text(x=-0.05, y=0.6, s=row_labels[1].upper(), fontsize='large')
# fig.text(x=-0.05, y=0.35, s=row_labels[2].upper(), fontsize='large')
# fig.text(x=-0.05, y=0.125, s=row_labels[3].upper(), fontsize='large')

# fig.savefig(plot_dir + 'composite_comp_son.png')
