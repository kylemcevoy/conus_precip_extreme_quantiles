import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import rioxarray
import xarray as xr

cvdp_dir = '/home/data/projects/conus_precip_extremes/cvdp/'
gpcc_dir = '/home/data/projects/conus_precip_extremes/gpcc/'
cesm2_dir = '/home/data/projects/conus_precip_extremes/cesm2/'
synthLE_dir = '/home/data/projects/conus_precip_extremes/synthLE/cesm2/'
obsle_dir = '/home/data/projects/conus_precip_extremes/obsLE/gpcc_cvdp/'
plot_dir = '/home/data/projects/conus_precip_extremes/plots/figures/'
shape_dir = '/home/data/projects/conus_precip_extremes/'

def find_wy_totals(da):
    nan_mask = ~da.isnull().any('time')
    water_year = da.time.dt.year + (da.time.dt.month >= 10)
    da = da.assign_coords({'water_year': water_year})
    wy_totals = da.groupby('water_year').sum('time')
    wy_totals = wy_totals.where(nan_mask)
    wy_totals = wy_totals.isel(water_year=slice(1, -1))
    return wy_totals

def calculate_wy_ts(da):
    cos_lat_weights = np.cos(np.deg2rad(da.lat))
    da_weighted = da.weighted(cos_lat_weights)
    da_ts = da_weighted.mean(['lat', 'lon'])
    return da_ts

def find_huc_ts(da, geom):
    da.rio.set_spatial_dims(x_dim='lon',
                            y_dim='lat',
                            inplace=True)
    da.rio.write_crs('wgs84', inplace=True)
    da_clipped = da.rio.clip([geom], crs='wgs84')
    
    da_totals = find_wy_totals(da_clipped)
    da_ts = calculate_wy_ts(da_totals)
    return da_ts

# def count_cons_dry(da, quantile, cons_years):
#     nan_mask = ~da.isnull().any('water_year')
#     if 'ens_mem' in da.dims:
#             dry_threshold = da.quantile(quantile, dim=['water_year', 'ens_mem'])
#     else:
#         dry_threshold = da.quantile(quantile, dim='water_year')
#     dry_years = (da < dry_threshold)
    
#     cumsum = dry_years.cumsum('water_year')
    
#     cumsum_masked = cumsum.where(~dry_years)
#     cumsum_masked = cumsum_masked.ffill('water_year')
#     cumsum_masked = cumsum_masked.fillna(0).where(nan_mask)
    
#     cons_dry_years = ((cumsum - cumsum_masked) == cons_years).sum('water_year')
    
#     return cons_dry_years

def max_cons_dry(da, quantile):
    nan_mask = ~da.isnull().any('water_year')
    if 'ens_mem' in da.dims:
            dry_threshold = da.quantile(quantile, dim=['water_year', 'ens_mem'])
    else:
        dry_threshold = da.quantile(quantile, dim='water_year')
    dry_years = (da < dry_threshold)
    
    cumsum = dry_years.cumsum('water_year')
    
    cumsum_masked = cumsum.where(~dry_years)
    cumsum_masked = cumsum_masked.ffill('water_year')
    cumsum_masked = cumsum_masked.fillna(0).where(nan_mask)
    
    max_cons_dry = ((cumsum - cumsum_masked)).max('water_year')
    
    return max_cons_dry

ortho_modes = xr.open_dataset(obsle_dir + 'ortho_modes.nc')
surr_modes = xr.open_dataset(obsle_dir + 'surrogate_modes.nc')

gpcc = xr.open_dataarray(gpcc_dir + 'gpcc_totals.nc')

## Shape 

with open(shape_dir + "huc2-basins.geojson") as json_file:
    data = json.load(json_file)

huc_basin_properties = [feat['properties'] for feat in data['features']]
huc_basin_names = [basin['name'] for basin in huc_basin_properties]
#Upper Colorado Region -- 9
#Lower Colorado Region -- 13
#California Region -- 12
#Great Basin Region -- 11
#Pacific Northwest Region -- 15

indx_list = [12, 15, 11, 9, 13]
name_list = [huc_basin_names[indx] for indx in indx_list]

geom_list = [feat['geometry'] for feat in data['features']]

geom_subset = [geom_list[indx] for indx in indx_list]

gpcc.rio.set_spatial_dims(x_dim='lon',
                          y_dim='lat',
                          inplace=True)

gpcc.rio.write_crs('wgs84', inplace=True)

gpcc_regions = [find_huc_ts(gpcc, geom) for geom in geom_subset]

gpcc_regions_dry = [count_cons_dry(gpcc_region, 0.5, 3) 
                     for gpcc_region in gpcc_regions]

gpcc_regions_max_dry = [max_cons_dry(gpcc_region, 0.5).values 
                     for gpcc_region in gpcc_regions]

cesm2 = xr.open_dataarray(cesm2_dir + 'cesm2_PRECT_processed.nc')
cesm2_totals = cesm2 * cesm2.time.dt.days_in_month
cesm2_totals = cesm2_totals.assign_coords({
    'lon': ((cesm2_totals.lon- 180) % 360) - 180
    })

cesm2_regions = [find_huc_ts(cesm2_totals, geom) for geom in geom_subset]

cesm2_regions_q20 = [cesm2_region.quantile(0.2, dim='water_year') 
                     for cesm2_region in cesm2_regions]

cesm2_regions_q80 = [cesm2_region.quantile(0.8, dim='water_year') 
                     for cesm2_region in cesm2_regions]

modes_paths = [synthLE_dir + f'mem{mem:02}/surrogate_modes.nc' 
               for mem in np.arange(50)]

synth_modes = xr.open_mfdataset(modes_paths,
                                combine='nested',
                                concat_dim='cesm2_mem')
synth_modes = synth_modes.compute()
synth_modes = synth_modes.assign_coords({'cesm2_mem': np.arange(50)})

water_year = synth_modes.time.dt.year + (synth_modes.time.dt.month >= 10)
synth_modes = synth_modes.assign_coords({'water_year': water_year})
modes_wy_means = synth_modes.groupby('water_year').mean('time')
modes_wy_means = modes_wy_means.sel(water_year=slice(1921, 2020))

ortho_modes_paths = [synthLE_dir + f'mem{mem:02}/ortho_modes.nc' 
               for mem in np.arange(50)]

ortho_modes = xr.open_mfdataset(ortho_modes_paths,
                                combine='nested',
                                concat_dim='cesm2_mem')
ortho_modes = ortho_modes.compute()
ortho_modes = ortho_modes.assign_coords({'cesm2_mem': np.arange(50)})
ortho_modes_wy = ortho_modes.time.dt.year + (ortho_modes.time.dt.month >= 10)
ortho_modes = ortho_modes.assign_coords({'water_year': ortho_modes_wy})
ortho_modes_wy_means = ortho_modes.groupby('water_year').mean('time')
ortho_modes_wy_means = ortho_modes_wy_means.sel(water_year=slice(1921, 2020))

water_year = synth_modes.time.dt.year + (synth_modes.time.dt.month >= 10)
synth_modes = synth_modes.assign_coords({'water_year': water_year})
modes_wy_means = synth_modes.groupby('water_year').mean('time')
modes_wy_means = modes_wy_means.sel(water_year=slice(1921, 2020))


# for i in range(50):
#     huc_ts_mem = xr.open_mfdataset(f'{synthLE_dir}analysis/synthLE_huc_ts_mem{i:02}_chunk*.nc')
#     huc_ts_mem.to_netcdf(f'{synthLE_dir}analysis/synthLE_huc_ts_mem{i:02}.nc')

# file_list = [f'{synthLE_dir}analysis/synthLE_huc_ts_mem{i:02}.nc' for i in range(50)]
# huc_ts_ens = xr.open_mfdataset(file_list,
#                                combine='nested',
#                                concat_dim='cesm2_mem')
# huc_ts_ens.to_netcdf(f'{synthLE_dir}analysis/synthLE_huc_ts_ens.nc')

# huc_ts_ens = huc_ts_ens.assign_coords({'ens_mem': np.arange(1, 1001)})

huc_ts_ens = xr.open_dataarray(f'{synthLE_dir}analysis/huc/synthLE_huc_ts_ens.nc')
huc_ts_ens = huc_ts_ens.assign_coords({"cesm2_mem": np.arange(50)})

huc_ts_ens = huc_ts_ens.transpose('region', 'water_year', 'cesm2_mem', 'ens_mem')

huc_ts_q20 = huc_ts_ens.quantile(0.2, dim='water_year')
huc_ts_q20_mean = huc_ts_q20.mean(dim='ens_mem')
huc_ts_q20_sd = huc_ts_q20.std(dim='ens_mem', ddof=1)
huc_ts_q20_mean_mean = huc_ts_q20_mean.mean(dim='cesm2_mem')
huc_ts_q20_sd_mean = huc_ts_q20_sd.mean(dim='cesm2_mem')

huc_ts_q80 = huc_ts_ens.quantile(0.8, dim='water_year')
huc_ts_q80_mean = huc_ts_q80.mean(dim='ens_mem')
huc_ts_q80_sd = huc_ts_q80.std(dim='ens_mem', ddof=1)
huc_ts_q80_mean_mean = huc_ts_q80_mean.mean(dim='cesm2_mem')
huc_ts_q80_sd_mean = huc_ts_q80_sd.mean(dim='cesm2_mem')

bins_mean_list = [np.linspace(450, 550, 11),
                  np.linspace(660, 740, 9),
                  np.linspace(350, 410, 7),
                  np.linspace(410, 480, 8),
                  np.linspace(260, 340, 9)]
bins_sd = np.linspace(5, 22, 18)

fig = plt.figure(layout='constrained', figsize=(12, 12))
subfigs = fig.subfigures(1, 2)
left_axs = subfigs[0].subplots(nrows=5)
right_axs = subfigs[1].subplots(nrows=5)
subfigs[0].suptitle('a) CESM2-Synth-LEs 0.2 quantile means')
subfigs[1].suptitle('b) CESM2-Synth-LEs 0.2 quantile standard deviations')
for i, region in enumerate(huc_ts_q20_sd.region):
    sns.histplot(huc_ts_q20_sd.sel(region=region), ax=right_axs[i], 
                 #bins=bins_sd
                 )
    right_axs[i].axvline(cesm2_regions_q20[i].std(ddof=1),
                color='red',
                linestyle='--')
    right_axs[i].axvline(huc_ts_q20_sd_mean.sel(region=name_list[i]),
                color='black',
                linestyle='--')
    right_axs[i].set_ylim((0, 25))
    right_axs[i].set_title(name_list[i])
    

    sns.histplot(huc_ts_q20_mean.sel(region=region), ax=left_axs[i], #bins=bins_mean_list[i])
    )
    left_axs[i].axvline(cesm2_regions_q20[i].mean(),
                color='red',
                linestyle='--')
    left_axs[i].axvline(huc_ts_q20_mean_mean.sel(region=name_list[i]),
                color='black',
                linestyle='--')
    left_axs[i].set_ylim((0, 25))
    left_axs[i].set_title(name_list[i])
left_axs[i].set_xlabel('mean 0.2 quantile [mm/water year]')
right_axs[i].set_xlabel('s.d. of 0.2 quantile [mm/water year]')
fig.savefig(plot_dir + 'huc_ts_q20_synth_validation.png')

fig = plt.figure(layout='constrained', figsize=(12, 12))
subfigs = fig.subfigures(1, 2)
left_axs = subfigs[0].subplots(nrows=5)
right_axs = subfigs[1].subplots(nrows=5)
subfigs[0].suptitle('a) CESM2-Synth-LEs 0.8 quantile means')
subfigs[1].suptitle('b) CESM2-Synth-LEs 0.8 quantile standard deviations')
for i, region in enumerate(huc_ts_q80_sd.region):
    sns.histplot(huc_ts_q80_sd.sel(region=region), ax=right_axs[i], 
                 #bins=bins_sd
                 )
    right_axs[i].axvline(cesm2_regions_q80[i].std(ddof=1),
                color='red',
                linestyle='--')
    right_axs[i].axvline(huc_ts_q80_sd_mean.sel(region=name_list[i]),
                color='black',
                linestyle='--')
    right_axs[i].set_ylim((0, 25))
    right_axs[i].set_title(name_list[i])
    

    sns.histplot(huc_ts_q80_mean.sel(region=region), ax=left_axs[i], #bins=bins_mean_list[i])
    )
    left_axs[i].axvline(cesm2_regions_q80[i].mean(),
                color='red',
                linestyle='--')
    left_axs[i].axvline(huc_ts_q80_mean_mean.sel(region=name_list[i]),
                color='black',
                linestyle='--')
    left_axs[i].set_ylim((0, 25))
    left_axs[i].set_title(name_list[i])
left_axs[i].set_xlabel('mean 0.8 quantile [mm/water year]')
right_axs[i].set_xlabel('s.d. of 0.8 quantile [mm/water year]')
fig.savefig(plot_dir + 'huc_ts_q80_synth_validation.png')


cesm2_regions_huc_ts = xr.concat(cesm2_regions, dim='region')
cesm2_regions_huc_ts = cesm2_regions_huc_ts.assign_coords(region=name_list)
cesm2_regions_huc_ts = cesm2_regions_huc_ts.rename({'ens_mem': 'cesm2_mem'})

cesm2_regions_enso_pos_q20 = (cesm2_regions_huc_ts.where(ortho_modes_wy_means['enso'] > 1)
                              .quantile(0.2, dim=['cesm2_mem', 'water_year']))
cesm2_regions_enso_neg_q20 = (cesm2_regions_huc_ts.where(ortho_modes_wy_means['enso'] > 1)
                              .quantile(0.2, dim=['cesm2_mem', 'water_year']))

synth_enso_pos_q20 = (huc_ts_ens.where(modes_wy_means['enso'] > 1)
                      .quantile(.2, dim='water_year'))

synth_enso_neg_q20 = (huc_ts_ens.where(modes_wy_means['enso'] < -1)
                      .quantile(.2, dim='water_year'))

synth_pos_q20_mean = synth_enso_pos_q20.mean(dim='ens_mem')
synth_pos_q20_mean_mean = synth_pos_q20_mean.mean(dim='cesm2_mem')
synth_neg_q20_mean = synth_enso_neg_q20.mean(dim='ens_mem')
synth_neg_q20_mean_mean = synth_neg_q20_mean.mean(dim='cesm2_mem')

synth_pos_q20_sd = synth_enso_pos_q20.std(dim='ens_mem', ddof=1)
synth_neg_q20_sd = synth_enso_neg_q20.std(dim='ens_mem', ddof=1)
synth_pos_q20_sd_mean = synth_pos_q20_sd.mean(dim='cesm2_mem')
synth_neg_q20_sd_mean = synth_neg_q20_sd.mean(dim='cesm2_mem')


cesm2_enso_pos_q20 = ((cesm2_regions_huc_ts.where(ortho_modes_wy_means['enso'] > 1))
                      .quantile(0.2, dim='water_year'))
cesm2_enso_neg_q20 = ((cesm2_regions_huc_ts.where(ortho_modes_wy_means['enso'] < -1))
                      .quantile(0.2, dim='water_year'))
cesm2_enso_pos_q20_mean = cesm2_enso_pos_q20.mean(dim='cesm2_mem')
cesm2_enso_neg_q20_mean = cesm2_enso_neg_q20.mean(dim='cesm2_mem')
cesm2_enso_pos_q20_sd = cesm2_enso_pos_q20.std(dim='cesm2_mem', ddof=1)
cesm2_enso_neg_q20_sd = cesm2_enso_neg_q20.std(dim='cesm2_mem', ddof=1)

cesm2_regions_enso_pos_q80 = (cesm2_regions_huc_ts.where(ortho_modes_wy_means['enso'] > 1)
                              .quantile(0.8, dim=['cesm2_mem', 'water_year']))
cesm2_regions_enso_neg_q80 = (cesm2_regions_huc_ts.where(ortho_modes_wy_means['enso'] > 1)
                              .quantile(0.8, dim=['cesm2_mem', 'water_year']))

synth_enso_pos_q80 = (huc_ts_ens.where(modes_wy_means['enso'] > 1)
                      .quantile(0.8, dim='water_year'))

synth_enso_neg_q80 = (huc_ts_ens.where(modes_wy_means['enso'] < -1)
                      .quantile(0.8, dim='water_year'))

synth_pos_q80_mean = synth_enso_pos_q80.mean(dim='ens_mem')
synth_pos_q80_mean_mean = synth_pos_q80_mean.mean(dim='cesm2_mem')
synth_neg_q80_mean = synth_enso_neg_q80.mean(dim='ens_mem')
synth_neg_q80_mean_mean = synth_neg_q80_mean.mean(dim='cesm2_mem')

synth_pos_q80_sd = synth_enso_pos_q80.std(dim='ens_mem', ddof=1)
synth_neg_q80_sd = synth_enso_neg_q80.std(dim='ens_mem', ddof=1)
synth_pos_q80_sd_mean = synth_pos_q80_sd.mean(dim='cesm2_mem')
synth_neg_q80_sd_mean = synth_neg_q80_sd.mean(dim='cesm2_mem')


cesm2_enso_pos_q80 = ((cesm2_regions_huc_ts.where(ortho_modes_wy_means['enso'] > 1))
                      .quantile(0.8, dim='water_year'))
cesm2_enso_neg_q80 = ((cesm2_regions_huc_ts.where(ortho_modes_wy_means['enso'] < -1))
                      .quantile(0.8, dim='water_year'))
cesm2_enso_pos_q80_mean = cesm2_enso_pos_q80.mean(dim='cesm2_mem')
cesm2_enso_neg_q80_mean = cesm2_enso_neg_q80.mean(dim='cesm2_mem')
cesm2_enso_pos_q80_sd = cesm2_enso_pos_q80.std(dim='cesm2_mem', ddof=1)
cesm2_enso_neg_q80_sd = cesm2_enso_neg_q80.std(dim='cesm2_mem', ddof=1)


fig = plt.figure(layout='constrained', figsize=(12, 12))
subfigs = fig.subfigures(1, 2)
left_axs = subfigs[0].subplots(nrows=5)
right_axs = subfigs[1].subplots(nrows=5)
subfigs[0].suptitle('ENSO Positive CESM2-Synth-LEs 0.2 quantile means')
subfigs[1].suptitle('ENSO Positive CESM2-Synth-LEs 0.2 quantile standard deviations')
for i, local_region in enumerate(synth_pos_q20_sd.region):
    sns.histplot(synth_pos_q20_sd.sel(region=local_region), ax=right_axs[i],
                 color='tab:green')
    right_axs[i].axvline(synth_pos_q20_sd_mean.sel(region=local_region),
                color='black',
                linestyle='--')
    right_axs[i].axvline(cesm2_enso_pos_q20_sd.sel(region=local_region),
                color='red',
                linestyle='--')
    right_axs[i].set_ylim((0, 25))
    right_axs[i].set_title(name_list[i])
    # if i in [2, 3]:
    #     right_axs[i].text(12.5, 23, name_list[i], horizontalalignment='left')
    # else:
    #     right_axs[i].text(5, 23, name_list[i], horizontalalignment='left')
    

    sns.histplot(synth_pos_q20_mean.sel(region=local_region), ax=left_axs[i],
                 color='tab:green') 
                 #bins=bins_mean_list[i])
    left_axs[i].axvline(cesm2_enso_pos_q20_mean.sel(region=local_region),
                color='red',
                linestyle='--')
    left_axs[i].axvline(synth_pos_q20_mean_mean.sel(region=local_region),
                color='black',
                linestyle='--')
    left_axs[i].set_ylim((0, 25))
    left_axs[i].set_title(name_list[i])
    #left_axs[i].text(bins_mean_list[i][-1], 23, name_list[i], horizontalalignment='right')

left_axs[i].set_xlabel('mean 0.2 quantile [mm/water year]')
right_axs[i].set_xlabel('s.d. of 0.2 quantile [mm/water year]')
fig.savefig(plot_dir + 'huc_q20_ts_synth_validation_enso_pos.png')
    
fig = plt.figure(layout='constrained', figsize=(12, 12))
subfigs = fig.subfigures(1, 2)
left_axs = subfigs[0].subplots(nrows=5)
right_axs = subfigs[1].subplots(nrows=5)
subfigs[0].suptitle('ENSO Negative CESM2-Synth-LEs 0.2 quantile means')
subfigs[1].suptitle('ENSO Negative CESM2-Synth-LEs 0.2 quantile standard deviations')
for i, local_region in enumerate(synth_neg_q20_sd.region):
    sns.histplot(synth_neg_q20_sd.sel(region=local_region), ax=right_axs[i],
                 color='tab:orange')
    right_axs[i].axvline(cesm2_enso_neg_q20_sd.sel(region=local_region),
                color='red',
                linestyle='--')
    right_axs[i].axvline(synth_neg_q20_sd_mean.sel(region=local_region),
                color='black',
                linestyle='--')

    right_axs[i].set_ylim((0, 25))
    right_axs[i].set_title(name_list[i])
    # if i in [2, 3]:
    #     right_axs[i].text(12.5, 23, name_list[i], horizontalalignment='left')
    # else:
    #     right_axs[i].text(5, 23, name_list[i], horizontalalignment='left')
    

    sns.histplot(synth_neg_q20_mean.sel(region=local_region), ax=left_axs[i],
                 color='tab:orange') 
                 #bins=bins_mean_list[i])
    left_axs[i].axvline(cesm2_enso_neg_q20_mean.sel(region=local_region),
                color='red',
                linestyle='--')
    left_axs[i].axvline(synth_neg_q20_mean_mean.sel(region=local_region),
                color='black',
                linestyle='--')
    left_axs[i].set_ylim((0, 25))
    left_axs[i].set_title(name_list[i])
    if i == 0:
        left_axs[i].set_xlim((470, 550))
    #left_axs[i].text(bins_mean_list[i][-1], 23, name_list[i], horizontalalignment='right')

left_axs[i].set_xlabel('mean 0.2 quantile [mm/water year]')
right_axs[i].set_xlabel('s.d. of 0.2 quantile [mm/water year]')
    
fig.savefig(plot_dir + 'huc_ts_synth_q20_validation_enso_neg.png')


fig = plt.figure(layout='constrained', figsize=(12, 12))
subfigs = fig.subfigures(1, 2)
left_axs = subfigs[0].subplots(nrows=5)
right_axs = subfigs[1].subplots(nrows=5)
subfigs[0].suptitle('ENSO Positive CESM2-Synth-LEs 0.8 quantile means')
subfigs[1].suptitle('ENSO Positive CESM2-Synth-LEs 0.8 quantile standard deviations')
for i, local_region in enumerate(synth_pos_q80_sd.region):
    sns.histplot(synth_pos_q80_sd.sel(region=local_region), ax=right_axs[i],
                 color='tab:green')
    right_axs[i].axvline(synth_pos_q80_sd_mean.sel(region=local_region),
                color='black',
                linestyle='--')
    right_axs[i].axvline(cesm2_enso_pos_q80_sd.sel(region=local_region),
                color='red',
                linestyle='--')
    right_axs[i].set_ylim((0, 25))
    right_axs[i].set_title(name_list[i])
    # if i in [2, 3]:
    #     right_axs[i].text(12.5, 23, name_list[i], horizontalalignment='left')
    # else:
    #     right_axs[i].text(5, 23, name_list[i], horizontalalignment='left')
    

    sns.histplot(synth_pos_q80_mean.sel(region=local_region), ax=left_axs[i],
                 color='tab:green') 
                 #bins=bins_mean_list[i])
    left_axs[i].axvline(cesm2_enso_pos_q80_mean.sel(region=local_region),
                color='red',
                linestyle='--')
    left_axs[i].axvline(synth_pos_q80_mean_mean.sel(region=local_region),
                color='black',
                linestyle='--')
    left_axs[i].set_ylim((0, 25))
    left_axs[i].set_title(name_list[i])
    #left_axs[i].text(bins_mean_list[i][-1], 23, name_list[i], horizontalalignment='right')

left_axs[i].set_xlabel('mean 0.8 quantile [mm/water year]')
right_axs[i].set_xlabel('s.d. of 0.8 quantile [mm/water year]')
fig.savefig(plot_dir + 'huc_q80_ts_synth_validation_enso_pos.png')


fig = plt.figure(layout='constrained', figsize=(12, 12))
subfigs = fig.subfigures(1, 2)
left_axs = subfigs[0].subplots(nrows=5)
right_axs = subfigs[1].subplots(nrows=5)
subfigs[0].suptitle('ENSO Negative CESM2-Synth-LEs 0.8 quantile means')
subfigs[1].suptitle('ENSO Negative CESM2-Synth-LEs 0.8 quantile standard deviations')
for i, local_region in enumerate(synth_neg_q80_sd.region):
    sns.histplot(synth_neg_q80_sd.sel(region=local_region), ax=right_axs[i],
                 color='tab:orange')
    right_axs[i].axvline(cesm2_enso_neg_q80_sd.sel(region=local_region),
                color='red',
                linestyle='--')
    right_axs[i].axvline(synth_neg_q80_sd_mean.sel(region=local_region),
                color='black',
                linestyle='--')

    right_axs[i].set_ylim((0, 25))
    right_axs[i].set_title(name_list[i])
    # if i in [2, 3]:
    #     right_axs[i].text(12.5, 23, name_list[i], horizontalalignment='left')
    # else:
    #     right_axs[i].text(5, 23, name_list[i], horizontalalignment='left')
    

    sns.histplot(synth_neg_q80_mean.sel(region=local_region), ax=left_axs[i],
                 color='tab:orange') 
                 #bins=bins_mean_list[i])
    left_axs[i].axvline(cesm2_enso_neg_q80_mean.sel(region=local_region),
                color='red',
                linestyle='--')
    left_axs[i].axvline(synth_neg_q80_mean_mean.sel(region=local_region),
                color='black',
                linestyle='--')
    left_axs[i].set_ylim((0, 25))
    left_axs[i].set_title(name_list[i])
    #left_axs[i].text(bins_mean_list[i][-1], 23, name_list[i], horizontalalignment='right')

left_axs[i].set_xlabel('mean 0.8 quantile [mm/water year]')
right_axs[i].set_xlabel('s.d. of 0.8 quantile [mm/water year]')
    
fig.savefig(plot_dir + 'huc_ts_synth_q80_validation_enso_neg.png')


synth_max_dry = max_cons_dry(huc_ts_ens, 0.5)

cesm2_max_dry = max_cons_dry(cesm2_regions_huc_ts, 0.5)

np.unique(synth_max_dry, axis=2, return_counts=True)

synth_max_dry.sel(region='California Region').to_pandas()

fig, ax = plt.subplots(nrows=5, ncols=1, dpi=400, figsize=(12, 8))
for i, region in enumerate(synth_max_dry.region):
    # sns.countplot(x=synth_max_dry.sel(region=region).values.flatten(),
    #           ax=ax[i],
    #           alpha=0.4,
    #           stat='proportion', 
    #           native_scale=True,
    #           label='synthetic')
    sns.countplot(x=cesm2_max_dry.sel(region=region).values.flatten(),
              ax=ax[0],
              stat='proportion',
              alpha=0.4,
              native_scale=True,
              label='cesm2')
    ax[i].set_xlim(0, 12)
    ax[i].set_ylim(0, 0.45)
    ax[i].set_title(name_list[i])
    
    
max_dry_values = np.arange(27)

counts_array = np.zeros((5, 50, 27))
for region in range(5):
    for cesm2_mem in range(50):
        for k, value in enumerate(max_dry_values):
            counts_array[region, cesm2_mem, k] = np.count_nonzero(synth_max_dry.isel(region=region,
                                                                                     cesm2_mem=cesm2_mem) == value)

proportions_array = counts_array / 1000

prop_quantiles = np.quantile(proportions_array, [0.05, 0.5, 0.95], axis=1)
upper_bar = prop_quantiles[2] - prop_quantiles[1]
lower_bar = prop_quantiles[1] - prop_quantiles[0]

cesm2_counts = np.zeros((5, 27))
for region in range(5):
    for k, value in enumerate(max_dry_values):
        cesm2_counts[region, k] = np.count_nonzero(cesm2_max_dry.values[region] == value)

cesm2_props = cesm2_counts / 50

fig.savefig(plot_dir + 'full_dist_max_dry.png')

fig, ax = plt.subplots(nrows=5, ncols=1, dpi=400, figsize=(12, 8))
for i, region in enumerate(synth_max_dry.region):
    ax[i].errorbar(x=max_dry_values, y=prop_quantiles[1, i], yerr=np.stack([upper_bar[i], lower_bar[i]], axis=0), fmt='o')
    sns.countplot(x=cesm2_max_dry.sel(region=region).values.flatten(),
              ax=ax[i],
              stat='proportion',
              alpha=0.4,
              native_scale=True,
              label='cesm2')
    ax[i].set_xlim(0, 12)
    ax[i].set_ylim(0, 0.45)
    ax[i].set_title(name_list[i])
    ax[i].scatter(6, 0.2)
fig.savefig(plot_dir + 'single_mem10_max_dry.png')

fig, ax = plt.subplots()
for i, region in enumerate(synth_max_dry.region):
    ax.errorbar(x=max_dry_values + (i + 1) * 0.2, y=prop_quantiles[1, i], yerr=np.stack([upper_bar[i], lower_bar[i]], axis=0), fmt='o')
    # for j in range(50):
    #     ax.scatter(max_dry_values, )


for i, value in enumerate(dry_values):
    for mem in np.arange(50):
        plt.scatter(value, dry_props[mem, i], alpha=0.3, color='tab:orange')
        plt.scatter(value, cesm2_props[])
ax.legend()



