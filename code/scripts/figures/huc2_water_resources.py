#import cartopy.feature as cf
#import cartopy.crs as ccrs
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
import seaborn as sns
import xarray as xr

gpcc_dir = '/home/data/projects/conus_precip_extremes/gpcc/'
cesm2_dir = '/home/data/projects/conus_precip_extremes/cesm2/'
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

def count_cons_dry(da, quantile, cons_years):
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
    
    cons_dry_years = ((cumsum - cumsum_masked) == cons_years).sum('water_year')
    
    return cons_dry_years

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

obsLE_totals = xr.open_dataarray(obsle_dir + 'obsLE_totals.nc')
obsLE_totals = obsLE_totals.drop_attrs()
obsLE_totals.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
obsLE_totals.rio.write_crs('wgs84', inplace=True)

obsLE_regions = [find_huc_ts(obsLE_totals, geom) for geom in geom_subset]
obsLE_regions_dry = [count_cons_dry(obsLE_region, 0.5, 3) 
                     for obsLE_region in obsLE_regions]
obsLE_regions_max_dry = [max_cons_dry(obsLE_region, 0.5)
                         for obsLE_region in obsLE_regions]

fig, ax = plt.subplots(figsize=(12, 12), nrows=2, ncols=2)

sns.histplot(obsLE_q20, ax=ax[0, 0], stat='density')
sns.histplot(obsLE_q80, ax=ax[1, 0], stat='density')

obsLE_count_dry = count_cons_dry(obsLE_ca_ts, 0.5, 3)

rng = np.random.default_rng(56)
binoms_05 = rng.binomial(1, 0.5, (1000, 100))

binom_verydry = binoms_05.astype('bool')
binom_da = xr.DataArray(data=binom_verydry, coords={'ens_mem': np.arange(1000),
                                                    'water_year': obsLE_ca_ts['water_year']})
cumsum = binom_da.cumsum('water_year')
cumsum_masked = cumsum.where(~binom_da)
cumsum_masked = cumsum_masked.ffill('water_year')
cumsum_masked = cumsum_masked.fillna(0)

surr_modes_water_year = surr_modes.time.dt.year + (surr_modes.time.dt.month >= 10)
surr_modes = surr_modes.assign_coords({'water_year': surr_modes_water_year})
surr_wy_means = surr_modes.groupby('water_year').mean('time')
surr_wy_means = surr_wy_means.sel(water_year=slice(1921, 2020))

modes_water_year = ortho_modes.time.dt.year + (ortho_modes.time.dt.month >= 10)
ortho_modes = ortho_modes.assign_coords({'water_year': modes_water_year})
modes_wy_means = ortho_modes.groupby('water_year').mean('time')
modes_wy_means = modes_wy_means.sel(water_year=slice(1921, 2020))

obsLE_q20_list = [(obsLE_region
                            .quantile(0.2, dim='water_year'))
                           for obsLE_region in obsLE_regions]

obsLE_q20_enso_neg_list = [(obsLE_region
                            .where(surr_wy_means['enso'] < -1)
                            .quantile(0.2, dim='water_year'))
                           for obsLE_region in obsLE_regions]
obsLE_q20_enso_pos_list = [(obsLE_region
                            .where(surr_wy_means['enso'] > 1)
                            .quantile(0.2, dim='water_year'))
                           for obsLE_region in obsLE_regions]

gpcc_q20_list = [(gpcc_region.quantile(0.2, dim='water_year').values)
                           for gpcc_region in gpcc_regions]

gpcc_q20_enso_neg_list = [(gpcc_region
                            .where(modes_wy_means['enso'] < -1)
                            .quantile(0.2, dim='water_year')).values
                           for gpcc_region in gpcc_regions]
gpcc_q20_enso_pos_list = [(gpcc_region
                            .where(modes_wy_means['enso'] > 1)
                            .quantile(0.2, dim='water_year')).values
                           for gpcc_region in gpcc_regions]

obsLE_q80_list = [(obsLE_region
                            .quantile(0.8, dim='water_year'))
                           for obsLE_region in obsLE_regions]

obsLE_q80_enso_neg_list = [(obsLE_region
                            .where(surr_wy_means['enso'] < -1)
                            .quantile(0.8, dim='water_year'))
                           for obsLE_region in obsLE_regions]
obsLE_q80_enso_pos_list = [(obsLE_region
                            .where(surr_wy_means['enso'] > 1)
                            .quantile(0.8, dim='water_year'))
                           for obsLE_region in obsLE_regions]

gpcc_q80_list = [(gpcc_region.quantile(0.8, dim='water_year').values)
                           for gpcc_region in gpcc_regions]

gpcc_q80_enso_neg_list = [(gpcc_region
                            .where(modes_wy_means['enso'] < -1)
                            .quantile(0.8, dim='water_year')).values
                           for gpcc_region in gpcc_regions]
gpcc_q80_enso_pos_list = [(gpcc_region
                            .where(modes_wy_means['enso'] > 1)
                            .quantile(0.8, dim='water_year')).values
                           for gpcc_region in gpcc_regions]

obsLE_q20_CI_lb = [obsLE_q20_region.quantile(0.025).values for obsLE_q20_region in obsLE_q20_list]
obsLE_q20_CI_ub = [obsLE_q20_region.quantile(0.975).values for obsLE_q20_region in obsLE_q20_list]
obsLE_q20_spread = [obsLE_q20_CI_ub[i] - obsLE_q20_CI_lb[i] for i in range(5)]

obsLE_q80_CI_lb = [obsLE_q80_region.quantile(0.025).values for obsLE_q80_region in obsLE_q80_list]
obsLE_q80_CI_ub = [obsLE_q80_region.quantile(0.975).values for obsLE_q80_region in obsLE_q80_list]
obsLE_q80_spread = [obsLE_q80_CI_ub[i] - obsLE_q80_CI_lb[i] for i in range(5)]

obsLE_q20_CI_lb_pos = [obsLE_q20_region.quantile(0.025).values for obsLE_q20_region in obsLE_q20_enso_pos_list]
obsLE_q20_CI_ub_pos = [obsLE_q20_region.quantile(0.975).values for obsLE_q20_region in obsLE_q20_enso_pos_list]
obsLE_q20_spread_pos = [obsLE_q20_CI_ub_pos[i] - obsLE_q20_CI_lb_pos[i] for i in range(5)]

obsLE_q20_CI_lb_neg = [obsLE_q20_region.quantile(0.025).values for obsLE_q20_region in obsLE_q20_enso_neg_list]
obsLE_q20_CI_ub_neg = [obsLE_q20_region.quantile(0.975).values for obsLE_q20_region in obsLE_q20_enso_neg_list]
obsLE_q20_spread_neg = [obsLE_q20_CI_ub_neg[i] - obsLE_q20_CI_lb_neg[i] for i in range(5)]

mpl.rcParams.update({'font.size': 16})
fig = plt.figure(layout='constrained', dpi=400, figsize=(12, 8))
subfigs = fig.subfigures(1, 2, wspace=0.07)
colors = ['tab:blue', 'tab:orange', 'tab:green']

axleft = subfigs[0].subplots(6, 1, 
                             gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 0.2]},
                             )

for i in range(5):
    sns.histplot(obsLE_q20_list[i].values,
             stat='density',
             ax=axleft[i],
             label='all years',
             alpha=0.4,
             color=colors[0],
             binwidth=10)
    sns.histplot(obsLE_q20_enso_neg_list[i].values, 
             ax=axleft[i],
             stat='density',
             label='negative ENSO',
             alpha=0.4,
             color=colors[1],
             binwidth=10)
    sns.histplot(obsLE_q20_enso_pos_list[i].values, 
             ax=axleft[i],
             stat='density',
             label='positive ENSO',
             alpha=0.4,
             color=colors[2],
             binwidth=10)
    axleft[i].set_xlim((150, 715))
    axleft[i].set_ylim((0, 0.055))
    axleft[i].axvline(gpcc_q20_list[i], linestyle='--', color=colors[0])
    axleft[i].axvline(gpcc_q20_enso_neg_list[i], 
                      linestyle='--',
                      color=colors[1])
    axleft[i].axvline(gpcc_q20_enso_pos_list[i], 
                    linestyle='--',
                    color=colors[2])
    axleft[i].set_title(name_list[i])

handles, labels = axleft[4].get_legend_handles_labels()
axleft[5].legend(handles, labels, ncols=3, loc='center')
axleft[5].axis('off')

subfigs[0].suptitle('HUC-2 Zone GPCC Synth-LE 0.2 Quantiles')

axright = subfigs[1].subplots(6, 1, 
                             gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 0.2]},
                             )

for i in range(5):
    sns.histplot(obsLE_q80_list[i].values,
             stat='density',
             ax=axright[i],
             label='all years',
             alpha=0.4,
             color=colors[0],
             binwidth=10)
    sns.histplot(obsLE_q80_enso_neg_list[i].values, 
             ax=axright[i],
             stat='density',
             label='negative ENSO',
             alpha=0.4,
             color=colors[1],
             binwidth=10)
    sns.histplot(obsLE_q80_enso_pos_list[i].values, 
             ax=axright[i],
             stat='density',
             label='positive ENSO',
             alpha=0.4,
             color=colors[2],
             binwidth=10)
    axright[i].set_xlim((150, 900))
    axright[i].set_ylim((0, 0.055))
    axright[i].axvline(gpcc_q80_list[i], linestyle='--', color=colors[0])
    axright[i].axvline(gpcc_q80_enso_neg_list[i], 
                      linestyle='--',
                      color=colors[1])
    axright[i].axvline(gpcc_q80_enso_pos_list[i], 
                    linestyle='--',
                    color=colors[2])
    axright[i].set_title(name_list[i])

handles, labels = axright[4].get_legend_handles_labels()
axright[5].legend(handles, labels, ncols=3, loc='center')
axright[5].axis('off')

axleft[4].set_xlabel('0.2 quantile [mm/water year]')
axright[4].set_xlabel('0.8 quantile [mm/water year]')

subfigs[1].suptitle('HUC-2 Zone GPCC Synth-LE 0.8 Quantiles')

# axright = subfigs[1].subplots(6, 1, gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 0.2]})
# axright[5].axis('off')

# for i in range(5):
#     sns.countplot(x=obsLE_regions_max_dry[i].values.astype('int'),
#               stat='proportion',
#               ax=axright[i],
#               alpha=0.4,
#               order=np.arange(1, 14))
#     axright[i].text(1.5, 0.30, name_list[i], horizontalalignment='left')
#     axright[i].set_ylim(0, 0.35)
#     axright[i].scatter(np.int64(gpcc_regions_max_dry[i] - 1), 0.05)
# axright[4].set_xlabel('maximum consecutive below median years')


fig.savefig(plot_dir + 'huc_zone_comp_q20_q80.png')




colors = ['tab:blue', 'tab:orange', 'tab:green']

mpl.rcParams.update({'font.size': 16})


fig = plt.figure(layout='constrained', dpi=400, figsize=(12, 12))
gs = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1, 0.2])

for i in range(5):
    ax_i = fig.add_subplot(gs[i, 0])
    sns.histplot(obsLE_q20_list[i].values,
             stat='density',
             ax=ax_i,
             label='all years',
             alpha=0.4,
             color=colors[0],
             binwidth=10)
    sns.histplot(obsLE_q20_enso_neg_list[i].values, 
             ax=ax_i,
             stat='density',
             label='negative ENSO',
             alpha=0.4,
             color=colors[1],
             binwidth=10)
    sns.histplot(obsLE_q20_enso_pos_list[i].values, 
             ax=ax_i,
             stat='density',
             label='positive ENSO',
             alpha=0.4,
             color=colors[2],
             binwidth=10)
    ax_i.set_xlim((150, 715))
    ax_i.set_ylim((0, 0.055))
    ax_i.axvline(gpcc_q20_list[i], linestyle='--', color=colors[0])
    ax_i.axvline(gpcc_q20_enso_neg_list[i], 
                      linestyle='--',
                      color=colors[1])
    ax_i.axvline(gpcc_q20_enso_pos_list[i], 
                    linestyle='--',
                    color=colors[2])
    ax_i.set_title(name_list[i])
    if i == 0:
        ax_i.set_title('a)', loc='left')

handles, labels = ax_i.get_legend_handles_labels()
ax_i.set_xlabel('0.2 quantile [mm/water year]')


# axleft[5].legend(handles, labels, ncols=3, loc='center')
# axleft[5].axis('off')

for i in range(5):
    ax_i = fig.add_subplot(gs[i, 1])
    sns.histplot(obsLE_q80_list[i].values,
             stat='density',
             ax=ax_i,
             label='all years',
             alpha=0.4,
             color=colors[0],
             binwidth=10)
    sns.histplot(obsLE_q80_enso_neg_list[i].values, 
             ax=ax_i,
             stat='density',
             label='negative ENSO',
             alpha=0.4,
             color=colors[1],
             binwidth=10)
    sns.histplot(obsLE_q80_enso_pos_list[i].values, 
             ax=ax_i,
             stat='density',
             label='positive ENSO',
             alpha=0.4,
             color=colors[2],
             binwidth=10)
    ax_i.set_xlim((150, 900))
    ax_i.set_ylim((0, 0.055))
    ax_i.axvline(gpcc_q80_list[i], linestyle='--', color=colors[0])
    ax_i.axvline(gpcc_q80_enso_neg_list[i], 
                      linestyle='--',
                      color=colors[1])
    ax_i.axvline(gpcc_q80_enso_pos_list[i], 
                    linestyle='--',
                    color=colors[2])
    ax_i.set_title(name_list[i])
    if i == 0:
        ax_i.set_title('b)', loc='left')

ax_i.set_xlabel('0.8 quantile [mm/water year]')

ax_legend = fig.add_subplot(gs[5, :])
ax_legend.legend(handles, labels, ncols=3, loc='center')
ax_legend.axis('off')

# fig.text(0.3, 1, '0.2 Quantiles',
#          transform=fig.transFigure, 
#          horizontalalignment='center',
#          fontsize=20)

# fig.text(0.81, 1, '0.8 Quantiles',
#          transform=fig.transFigure, 
#          horizontalalignment='center',
#          fontsize=20)

fig.savefig(plot_dir + 'huc_zone_comp_q20_q80.png')
