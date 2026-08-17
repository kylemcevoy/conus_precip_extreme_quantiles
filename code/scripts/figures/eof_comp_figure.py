import cartopy.crs as ccrs
import cartopy.feature as cf
import eofs.xarray as eof
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from string import ascii_lowercase

mpl.rcParams.update({'font.size': 16})

def find_stacked_eofs_monthly(ensemble_da, month):
    import calendar
    
    month_name = calendar.month_abbr[month].upper()
    start_date = f'1920-{month}-01'
    
    ensemble_da = ensemble_da.sel(time=ensemble_da['time.month'] == month)
    #ensure that the dimensions are in the proper order for when we pass to
    # numpy
    ensemble_da = ensemble_da.transpose('ens_mem', 'time', 'lat', 'lon')
    ens_mems = ensemble_da.ens_mem
    
    ensemble_list = []
    for mem in ens_mems:
        ensemble_list.append(ensemble_da.sel(ens_mem=mem).values)
        
    ensemble_stacked = np.vstack(ensemble_list)
    samples = ensemble_stacked.shape[0]
    
    fake_time = xr.date_range(start=start_date,
                              periods=samples,
                              freq=f'YS-{month_name}',
                              calendar='noleap')
    
    ensemble_stacked_da = xr.DataArray(data=ensemble_stacked,
                                       coords={'time':fake_time,
                                               'lat': ensemble_da.lat,
                                               'lon': ensemble_da.lon})
    
    # get monthly anomalies (only one month left)
    ensemble_stacked_da = (ensemble_stacked_da 
                           - ensemble_stacked_da.mean('time'))
    
    cos_lats = np.cos(np.deg2rad(ensemble_stacked_da['lat'])).values
    cos_weights = np.sqrt(cos_lats)[..., np.newaxis]
    
    eof_solver = eof.Eof(ensemble_stacked_da, weights=cos_weights)
    
    #normalized to L2 norm = 1
    eofs = eof_solver.eofs(eofscaling=0)
    PC_ts = eof_solver.pcs()
    percent_var = eof_solver.varianceFraction().values
    
    return eofs, PC_ts, percent_var

def find_stacked_eofs_seasonal(ensemble_da, season):
    import calendar
    
    season_to_month = {'DJF': 12,
                       'MAM': 3,
                       'JJA': 6,
                       'SON': 9}
    
    start_month = season_to_month[season]
    month_name = calendar.month_abbr[start_month].upper()
    
    # don't want partial DJFs
    if season == 'DJF':
        ensemble_da = ensemble_da.sel(time=slice('1920-03-01', '2020-11-01'))
    
    start_date = f'1920-{start_month:02}-01'
    
    ensemble_da = ensemble_da.convert_calendar('noleap')
    nan_mask = ~(ensemble_da.isnull().any('time'))
    days_in_month = ensemble_da.time.dt.days_in_month
        
    seasonal_sums = ((ensemble_da * days_in_month)
                        .resample(time='QS-DEC')
                        .sum('time')
                        .where(nan_mask))
    
    days_in_season = (days_in_month.resample(time='QS-DEC')
                      .sum('time')
                      .where(nan_mask))
    
    seasonal_averages = seasonal_sums / days_in_season
    seasonal_averages = (seasonal_averages.sel(time=seasonal_averages['time.season'] == season))
   
    seasonal_sums = seasonal_sums.sel(time=seasonal_sums['time.season'] == season)

    #ensure that the dimensions are in the proper order for when we pass to
    # numpy
    seasonal_sums = seasonal_sums.transpose('ens_mem', 'time', 'lat', 'lon')
    ens_mems = seasonal_sums.ens_mem
        
    ensemble_list = []
    average_list = []
    for mem in ens_mems:
        ensemble_list.append(seasonal_sums.sel(ens_mem=mem).values)
        average_list.append(seasonal_averages.sel(ens_mem=mem).values)
        
    ensemble_stacked = np.vstack(ensemble_list)
    average_stacked = np.vstack(average_list)
    samples = ensemble_stacked.shape[0]
    
    fake_time = xr.date_range(start=start_date,
                              periods=samples,
                              freq=f'YS-{month_name}',
                              calendar='noleap')
    
    ensemble_stacked_da = xr.DataArray(data=ensemble_stacked,
                                       coords={'time':fake_time,
                                               'lat': ensemble_da.lat,
                                               'lon': ensemble_da.lon})
    
    average_stacked_da = xr.DataArray(data=average_stacked,
                                       coords={'time':fake_time,
                                               'lat': ensemble_da.lat,
                                               'lon': ensemble_da.lon})
    
    # get seasonal anomalies (only one season left)
    ensemble_stacked_da = (ensemble_stacked_da 
                           - ensemble_stacked_da.mean('time'))
    
    cos_lats = np.cos(np.deg2rad(ensemble_stacked_da['lat'])).values
    cos_weights = np.sqrt(cos_lats)[..., np.newaxis]
    
    eof_solver = eof.Eof(ensemble_stacked_da, weights=cos_weights)
    
    #normalized to L2 norm = 1
    eofs = eof_solver.eofs(eofscaling=0)
    PC_ts = eof_solver.pcs()
    percent_var = eof_solver.varianceFraction().values
    total_var = ensemble_stacked_da.var('time', ddof=1)
    
    return eofs, total_var, PC_ts, percent_var

proj_dir = '/home/data/projects/conus_precip_extremes/'
synthLE_dir = 'synthLE/cesm2/mem00/'
obsLE_dir = 'obsLE/gpcc_cvdp/'

# first 50 members of GPCC synthLE

gpcc_file_list = [proj_dir + obsLE_dir + f'obsLE_member{i + 1:04}.nc' for i in np.arange(50)]

gpcc_obsLE = xr.open_mfdataset(gpcc_file_list,
                               combine='nested',
                               concat_dim='ens_mem',
                               )

gpcc_obsLE = gpcc_obsLE.compute()
gpcc_obsLE = gpcc_obsLE['precip']
gpcc_obsLE = gpcc_obsLE.assign_coords({'ens_mem': np.arange(50)})

cesm2_le = xr.open_dataarray('/home/data/projects/conus_precip_extremes/cesm2/cesm2_PRECT_processed.nc')

file_list = [proj_dir + synthLE_dir + f'obsLE_member{i + 1:04}.nc' for i in np.arange(50)]
cesm2_synthLE = xr.open_mfdataset(file_list,
                               combine='nested',
                               concat_dim='synth_ens_mem')

cesm2_synthLE = cesm2_synthLE.compute()
cesm2_synthLE = cesm2_synthLE['precip']
cesm2_synthLE = cesm2_synthLE.drop_vars('ens_mem')
cesm2_synthLE = cesm2_synthLE.rename({'synth_ens_mem': 'ens_mem'})

#### Dec.

gpcc_eofs = find_stacked_eofs_monthly(gpcc_obsLE, month=12)
gpcc_eofs_modes = gpcc_eofs[0]
gpcc_per_var = gpcc_eofs[2]

cmap = mpl.cm.BrBG
levels = [-0.2, -0.1, -0.05, -0.025, 0, 0.025, 0.05, 0.1, 0.2]
norm = mpl.colors.BoundaryNorm(levels, cmap.N, extend='both')


fig, ax = plt.subplots(nrows=4,
                       constrained_layout=True,
                       sharex=True,
                       subplot_kw={'projection': ccrs.PlateCarree()})
for i in range(4):
    gpcc_eofs_modes.sel(mode = i).plot(ax=ax[i],
                                   cmap=cmap, 
                                   norm=norm,
                                   add_colorbar=False)
    ax[i].set_title(f'GPCC SynthLE Dec. EoF {i + 1}')
    ax[i].coastlines()
    ax[i].add_feature(cf.BORDERS)
fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
             label='precip anom. [mm/day]')
plt.close('all')

cesm2le_eofs = find_stacked_eofs_monthly(cesm2_le, month=12)
cesm2le_eof_modes = cesm2le_eofs[0]
cesm2le_per_var = cesm2le_eofs[2]

cesm2_synthLE_eofs = find_stacked_eofs_monthly(cesm2_synthLE, month=12)
cesm2_synthLE_eof_modes = cesm2_synthLE_eofs[0]
cesm2_synthLE_eof_modes.loc[{'mode': 0}] = -1 * cesm2_synthLE_eof_modes.loc[{'mode': 0}]
cesm2_synthLE_per_var = cesm2_synthLE_eofs[2]

ens_names = ['CESM2', 'CESM2 Synth', 'GPCC Synth']
ens_eof_list = [cesm2le_eof_modes, cesm2_synthLE_eof_modes, gpcc_eofs_modes]
per_var_list = [cesm2le_per_var, cesm2_synthLE_per_var, gpcc_per_var]

fig, ax = plt.subplots(figsize=(14, 8),
                       ncols=3,
                       nrows=4,
                       dpi=400,
                       constrained_layout=True,
                       subplot_kw={'projection': ccrs.PlateCarree()})

# fig = plt.figure()
# gs = fig.add_gridspec(4, 3, hspace=0, wspace=0, width_ratios=[8, 8, 8],
#                       *{"projection": ccrs.PlateCarree()})
# a = gs.subplots(sharex='col')
for j in range(3):
    for i in range(4):
        ens_eof_list[j].sel(mode = i).plot(ax=ax[i, j],
                                    cmap=cmap, 
                                    norm=norm,
                                    add_colorbar=False)
        
        ax[i, j].set_title('')
        
        ax_index = 4 * j + (i)
        ax_label = ascii_lowercase[ax_index]
        
        per_var = per_var_list[j][i]
        ax[i, j].set_title(f'({ax_label}) {ens_names[j]} EOF {i + 1} ({per_var * 100:.1f}% var. exp.)', 
                           fontsize=12,
                           loc='left')
        
        ax[i, j].coastlines()
        ax[i, j].add_feature(cf.BORDERS)
fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
             label='precip anom. [mm/day]',
             shrink=0.8, pad=0.02)

fig.savefig('/home/data/projects/conus_precip_extremes/plots/figures/eof_comp_dec.png')

plt.close('all')



### DJF EOF comp

cmap = mpl.cm.BrBG
levels = [-0.2, -0.1, -0.05, -0.025, 0, 0.025, 0.05, 0.1, 0.2]
norm = mpl.colors.BoundaryNorm(levels, cmap.N, extend='both')

var_cmap = mpl.cm.viridis
var_levels = np.array([0.0, 1500, 3000, 6000, 12000])
var_norm = mpl.colors.BoundaryNorm(var_levels, var_cmap.N, extend='max')

gpcc_djf_eof_output = find_stacked_eofs_seasonal(gpcc_obsLE, season='DJF')
gpcc_djf_eofs = gpcc_djf_eof_output[0]
gpcc_djf_total_var = gpcc_djf_eof_output[1]
gpcc_djf_per_var = gpcc_djf_eof_output[3]

cesm2le_djf_eof_output = find_stacked_eofs_seasonal(cesm2_le, season='DJF')
cesm2le_djf_eofs = cesm2le_djf_eof_output[0]
cesm2le_djf_total_var = cesm2le_djf_eof_output[1]
cesm2le_djf_per_var = cesm2le_djf_eof_output[3]

cesm2_synthLE_djf_eof_output = find_stacked_eofs_seasonal(cesm2_synthLE, 
                                                          season='DJF')
cesm2_synthLE_djf_eofs = cesm2_synthLE_djf_eof_output[0]
# signs are arbitrary so doing some sign matching
cesm2_synthLE_djf_eofs.loc[{'mode': 1}] = -1 * cesm2_synthLE_djf_eofs.loc[{'mode': 1}]
cesm2_synthLE_djf_eofs.loc[{'mode': 3}] = -1 * cesm2_synthLE_djf_eofs.loc[{'mode': 3}]
cesm2_synthLE_djf_total_var = cesm2_synthLE_djf_eof_output[1]
cesm2_synthLE_djf_per_var = cesm2_synthLE_djf_eof_output[3]

ens_names = ['CESM2', 'CESM2-synth', 'GPCC-synth']
ens_djf_eof_list = [cesm2le_djf_eofs, cesm2_synthLE_djf_eofs, gpcc_djf_eofs]
per_var_djf_list = [cesm2le_djf_per_var, cesm2_synthLE_djf_per_var, gpcc_djf_per_var]
total_var_list = [cesm2le_djf_total_var, 
                  cesm2_synthLE_djf_total_var,
                  gpcc_djf_total_var]

fig = plt.figure(figsize=(16, 12), constrained_layout=True, dpi=400)
gs = gridspec.GridSpec(5, 4, figure=fig, width_ratios=[1, 1, 1, 0.05])

for j in range(3):
    for i in range(5):
        ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
        if i == 0:
            total_var_list[j].plot(ax=ax,
                                   cmap=var_cmap,
                                   norm=var_norm,
                                   add_colorbar=False)
            ax.set_title('')
            
            ax_index = 5 * j + (i)
            ax_label = ascii_lowercase[ax_index]
            
            ax.set_title(f'({ax_label}) {ens_names[j]} variance',
                               loc='left')
            ax.coastlines()
            ax.add_feature(cf.BORDERS)
        else:
            ens_djf_eof_list[j].sel(mode = i - 1).plot(ax=ax,
                                        cmap=cmap, 
                                        norm=norm,
                                        add_colorbar=False)
            
            ax.set_title('')
            
            ax_index = 5 * j + (i)
            ax_label = ascii_lowercase[ax_index]
            
            per_var = per_var_djf_list[j][i]
            ax.set_title(f'({ax_label}) {ens_names[j]} EOF {i} ({per_var * 100:.1f}%)', 

                            loc='left')
            
            ax.coastlines()
            ax.add_feature(cf.BORDERS)

cbar1 = fig.colorbar(cm.ScalarMappable(cmap=var_cmap, norm=var_norm), 
             cax=plt.subplot(gs[0, 3]),
             shrink=0.7, 
             pad=0.02)

cbar1.set_label(r'var. [$\mathrm{mm}^2$]', fontsize=20)
cbar1.ax.tick_params(labelsize=20)

cbar2 = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
             cax=plt.subplot(gs[1:, 3]),
             
             shrink=0.5, pad=0.02)

cbar2.set_label('Precipitation EOFs [unitless]',
                fontsize=20)
cbar2.ax.tick_params(labelsize=20)

# fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
#              cax=plt.subplot(gs[2, 3]),
#              label='Precip. EOFs [ ]',
#              shrink=0.7, pad=0.02)

# fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
#              cax=plt.subplot(gs[3, 3]),
#              label='Precip. EOFs [ ]',
#              shrink=0.7, pad=0.02)

# fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
#              cax=plt.subplot(gs[4, 3]),
#              label='Precip. EOFs [ ]',
#              shrink=0.7, pad=0.02)

fig.savefig('/home/data/projects/conus_precip_extremes/plots/figures/eof_comp_djf.png')

plt.close('all')



### MAM EOF comp

cmap = mpl.cm.BrBG
levels = [-0.2, -0.1, -0.05, -0.025, 0, 0.025, 0.05, 0.1, 0.2]
norm = mpl.colors.BoundaryNorm(levels, cmap.N, extend='both')

var_cmap = mpl.cm.viridis
var_levels = [0.0, 0.125, 0.250, 0.500, 1.0, 2.0]
var_norm = mpl.colors.BoundaryNorm(var_levels, var_cmap.N, extend='max')

gpcc_mam_eof_output = find_stacked_eofs_seasonal(gpcc_obsLE, season='MAM')
gpcc_mam_eofs = gpcc_mam_eof_output[0]
gpcc_mam_total_var = gpcc_mam_eof_output[1]
gpcc_mam_per_var = gpcc_mam_eof_output[3]

cesm2le_mam_eof_output = find_stacked_eofs_seasonal(cesm2_le, season='MAM')
cesm2le_mam_eofs = cesm2le_mam_eof_output[0]
cesm2le_mam_total_var = cesm2le_mam_eof_output[1]
cesm2le_mam_per_var = cesm2le_mam_eof_output[3]

cesm2_synthLE_mam_eof_output = find_stacked_eofs_seasonal(cesm2_synthLE, 
                                                          season='MAM')
cesm2_synthLE_mam_eofs = cesm2_synthLE_mam_eof_output[0]
# signs are arbitrary so doing some sign matching
cesm2_synthLE_mam_eofs.loc[{'mode': 1}] = -1 * cesm2_synthLE_mam_eofs.loc[{'mode': 1}]
cesm2_synthLE_mam_eofs.loc[{'mode': 3}] = -1 * cesm2_synthLE_mam_eofs.loc[{'mode': 3}]
cesm2_synthLE_mam_total_var = cesm2_synthLE_mam_eof_output[1]
cesm2_synthLE_mam_per_var = cesm2_synthLE_mam_eof_output[3]

ens_names = ['CESM2', 'CESM2 Synth', 'GPCC Synth']
ens_mam_eof_list = [cesm2le_mam_eofs, cesm2_synthLE_mam_eofs, gpcc_mam_eofs]
per_var_mam_list = [cesm2le_mam_per_var, cesm2_synthLE_mam_per_var, gpcc_mam_per_var]
total_var_list = [cesm2le_mam_total_var, 
                  cesm2_synthLE_mam_total_var,
                  gpcc_mam_total_var]

fig = plt.figure(figsize=(16, 12), constrained_layout=True, dpi=400)
gs = gridspec.GridSpec(5, 4, figure=fig, width_ratios=[1, 1, 1, 0.05])

for j in range(3):
    for i in range(5):
        ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
        if i == 0:
            total_var_list[j].plot(ax=ax,
                                   cmap=var_cmap,
                                   norm=var_norm,
                                   add_colorbar=False)
            ax.set_title('')
            
            ax_index = 4 * j + (i)
            ax_label = ascii_lowercase[ax_index]
            
            ax.set_title(f'({ax_label}) {ens_names[j]} precipitation variance)', 
                               fontsize=12,
                               loc='left')
            ax.coastlines()
            ax.add_feature(cf.BORDERS)
        else:
            ens_mam_eof_list[j].sel(mode = i - 1).plot(ax=ax,
                                        cmap=cmap, 
                                        norm=norm,
                                        add_colorbar=False)
            
            ax.set_title('')
            
            ax_index = 4 * j + (i)
            ax_label = ascii_lowercase[ax_index]
            
            per_var = per_var_mam_list[j][i]
            ax.set_title(f'({ax_label}) {ens_names[j]} EOF {i} ({per_var * 100:.1f}% var. exp.)', 
                            fontsize=12,
                            loc='left')
            
            ax.coastlines()
            ax.add_feature(cf.BORDERS)

fig.colorbar(cm.ScalarMappable(cmap=var_cmap, norm=var_norm), 
             cax=plt.subplot(gs[0, 3]),
             label=r'Precip. var. [$(\mathrm{mm/day})^2$]',
             shrink=0.7, pad=0.02)

fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
             cax=plt.subplot(gs[1, 3]),
             label='Precip. EOFs [ ]',
             shrink=0.7, pad=0.02)

fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
             cax=plt.subplot(gs[2, 3]),
             label='Precip. EOFs [ ]',
             shrink=0.7, pad=0.02)

fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
             cax=plt.subplot(gs[3, 3]),
             label='Precip. EOFs [ ]',
             shrink=0.7, pad=0.02)

fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
             cax=plt.subplot(gs[4, 3]),
             label='Precip. EOFs [ ]',
             shrink=0.7, pad=0.02)

fig.savefig('/home/data/projects/conus_precip_extremes/plots/figures/eof_comp_mam.png')

### JJA EOF comp

gpcc_jja_eof_output = find_stacked_eofs_seasonal(gpcc_obsLE, season='JJA')
gpcc_jja_eofs = gpcc_jja_eof_output[0]
gpcc_jja_total_var = gpcc_jja_eof_output[1]
gpcc_jja_per_var = gpcc_jja_eof_output[3]

cesm2le_jja_eof_output = find_stacked_eofs_seasonal(cesm2_le, season='JJA')
cesm2le_jja_eofs = cesm2le_jja_eof_output[0]
cesm2le_jja_total_var = cesm2le_jja_eof_output[1]
cesm2le_jja_per_var = cesm2le_jja_eof_output[3]

cesm2_synthLE_jja_eof_output = find_stacked_eofs_seasonal(cesm2_synthLE, 
                                                          season='JJA')
cesm2_synthLE_jja_eofs = cesm2_synthLE_jja_eof_output[0]
# signs are arbitrary so doing some sign matching
cesm2_synthLE_jja_eofs.loc[{'mode': 1}] = -1 * cesm2_synthLE_jja_eofs.loc[{'mode': 1}]
cesm2_synthLE_jja_eofs.loc[{'mode': 3}] = -1 * cesm2_synthLE_jja_eofs.loc[{'mode': 3}]
cesm2_synthLE_jja_total_var = cesm2_synthLE_jja_eof_output[1]
cesm2_synthLE_jja_per_var = cesm2_synthLE_jja_eof_output[3]

ens_names = ['CESM2', 'CESM2 Synth', 'GPCC Synth']
ens_jja_eof_list = [cesm2le_jja_eofs, cesm2_synthLE_jja_eofs, gpcc_jja_eofs]
per_var_jja_list = [cesm2le_jja_per_var, cesm2_synthLE_jja_per_var, gpcc_jja_per_var]
total_var_list = [cesm2le_jja_total_var, 
                  cesm2_synthLE_jja_total_var,
                  gpcc_jja_total_var]

fig = plt.figure(figsize=(16, 12), constrained_layout=True, dpi=400)
gs = gridspec.GridSpec(5, 4, figure=fig, width_ratios=[1, 1, 1, 0.05])

for j in range(3):
    for i in range(5):
        ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
        if i == 0:
            total_var_list[j].plot(ax=ax,
                                   cmap=var_cmap,
                                   norm=var_norm,
                                   add_colorbar=False)
            ax.set_title('')
            
            ax_index = 4 * j + (i)
            ax_label = ascii_lowercase[ax_index]
            
            ax.set_title(f'({ax_label}) {ens_names[j]} precipitation variance)', 
                               fontsize=12,
                               loc='left')
            ax.coastlines()
            ax.add_feature(cf.BORDERS)
        else:
            ens_jja_eof_list[j].sel(mode = i - 1).plot(ax=ax,
                                        cmap=cmap, 
                                        norm=norm,
                                        add_colorbar=False)
            
            ax.set_title('')
            
            ax_index = 4 * j + (i)
            ax_label = ascii_lowercase[ax_index]
            
            per_var = per_var_jja_list[j][i]
            ax.set_title(f'({ax_label}) {ens_names[j]} EOF {i} ({per_var * 100:.1f}% var. exp.)', 
                            fontsize=12,
                            loc='left')
            
            ax.coastlines()
            ax.add_feature(cf.BORDERS)

fig.colorbar(cm.ScalarMappable(cmap=var_cmap, norm=var_norm), 
             cax=plt.subplot(gs[0, 3]),
             label=r'Precip. var. [$(\mathrm{mm/day})^2$]',
             shrink=0.7, pad=0.02)

fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
             cax=plt.subplot(gs[1, 3]),
             label='Precip. EOFs [ ]',
             shrink=0.7, pad=0.02)

fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
             cax=plt.subplot(gs[2, 3]),
             label='Precip. EOFs [ ]',
             shrink=0.7, pad=0.02)

fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
             cax=plt.subplot(gs[3, 3]),
             label='Precip. EOFs [ ]',
             shrink=0.7, pad=0.02)

fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
             cax=plt.subplot(gs[4, 3]),
             label='Precip. EOFs [ ]',
             shrink=0.7, pad=0.02)

fig.savefig('/home/data/projects/conus_precip_extremes/plots/figures/eof_comp_jja.png')

### SON EOF comp

gpcc_son_eof_output = find_stacked_eofs_seasonal(gpcc_obsLE, season='SON')
gpcc_son_eofs = gpcc_son_eof_output[0]
gpcc_son_total_var = gpcc_son_eof_output[1]
gpcc_son_per_var = gpcc_son_eof_output[3]

cesm2le_son_eof_output = find_stacked_eofs_seasonal(cesm2_le, season='SON')
cesm2le_son_eofs = cesm2le_son_eof_output[0]
cesm2le_son_total_var = cesm2le_son_eof_output[1]
cesm2le_son_per_var = cesm2le_son_eof_output[3]

cesm2_synthLE_son_eof_output = find_stacked_eofs_seasonal(cesm2_synthLE, 
                                                          season='SON')
cesm2_synthLE_son_eofs = cesm2_synthLE_son_eof_output[0]
# signs are arbitrary so doing some sign matching
cesm2_synthLE_son_eofs.loc[{'mode': 1}] = -1 * cesm2_synthLE_son_eofs.loc[{'mode': 1}]
cesm2_synthLE_son_eofs.loc[{'mode': 3}] = -1 * cesm2_synthLE_son_eofs.loc[{'mode': 3}]
cesm2_synthLE_son_total_var = cesm2_synthLE_son_eof_output[1]
cesm2_synthLE_son_per_var = cesm2_synthLE_son_eof_output[3]

ens_names = ['CESM2', 'CESM2 Synth', 'GPCC Synth']
ens_son_eof_list = [cesm2le_son_eofs, cesm2_synthLE_son_eofs, gpcc_son_eofs]
per_var_son_list = [cesm2le_son_per_var, cesm2_synthLE_son_per_var, gpcc_son_per_var]
total_var_list = [cesm2le_son_total_var, 
                  cesm2_synthLE_son_total_var,
                  gpcc_son_total_var]

fig = plt.figure(figsize=(16, 12), constrained_layout=True, dpi=400)
gs = gridspec.GridSpec(5, 4, figure=fig, width_ratios=[1, 1, 1, 0.05])

for j in range(3):
    for i in range(5):
        ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
        if i == 0:
            total_var_list[j].plot(ax=ax,
                                   cmap=var_cmap,
                                   norm=var_norm,
                                   add_colorbar=False)
            ax.set_title('')
            
            ax_index = 4 * j + (i)
            ax_label = ascii_lowercase[ax_index]
            
            ax.set_title(f'({ax_label}) {ens_names[j]} precipitation variance)', 
                               fontsize=12,
                               loc='left')
            ax.coastlines()
            ax.add_feature(cf.BORDERS)
        else:
            ens_son_eof_list[j].sel(mode = i - 1).plot(ax=ax,
                                        cmap=cmap, 
                                        norm=norm,
                                        add_colorbar=False)
            
            ax.set_title('')
            
            ax_index = 4 * j + (i)
            ax_label = ascii_lowercase[ax_index]
            
            per_var = per_var_son_list[j][i]
            ax.set_title(f'({ax_label}) {ens_names[j]} EOF {i} ({per_var * 100:.1f}% var. exp.)', 
                            fontsize=12,
                            loc='left')
            
            ax.coastlines()
            ax.add_feature(cf.BORDERS)

fig.colorbar(cm.ScalarMappable(cmap=var_cmap, norm=var_norm), 
             cax=plt.subplot(gs[0, 3]),
             label=r'Precip. var. [$(\mathrm{mm/day})^2$]',
             shrink=0.7, pad=0.02)

fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
             cax=plt.subplot(gs[1, 3]),
             label='Precip. EOFs [ ]',
             shrink=0.7, pad=0.02)

fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
             cax=plt.subplot(gs[2, 3]),
             label='Precip. EOFs [ ]',
             shrink=0.7, pad=0.02)

fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
             cax=plt.subplot(gs[3, 3]),
             label='Precip. EOFs [ ]',
             shrink=0.7, pad=0.02)

fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
             cax=plt.subplot(gs[4, 3]),
             label='Precip. EOFs [ ]',
             shrink=0.7, pad=0.02)
fig.savefig('/home/data/projects/conus_precip_extremes/plots/figures/eof_comp_son.png')

#flip cesm2 1, 4
#flip cesm2 synth 2
# flip gpcc 2
