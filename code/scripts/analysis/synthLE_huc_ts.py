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
output_dir = synthLE_dir + 'analysis/'
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

chunk_mems = [np.arange((j * 100) + 1, (j + 1) * 100 + 1) for j in range(10)]

for ens_mem in range(50):
    print(ens_mem)
    data_dir = f'{synthLE_dir}mem{ens_mem:02}/'
    for i, chunk in enumerate(chunk_mems):
        print(i)
        file_names = [f'{data_dir}obsLE_member{mem:04}.nc' for mem in chunk]
        chunk_data = xr.open_mfdataset(file_names,
                                    combine='nested',
                                    concat_dim='ens_mem')
        chunk_data = chunk_data.compute()
        chunk_data = chunk_data.time.dt.days_in_month * chunk_data
        chunk_data = chunk_data.assign_coords({'ens_mem': chunk})
        chunk_data = chunk_data.drop_attrs()
        chunk_data.rio.set_spatial_dims(x_dim='lon',
                            y_dim='lat',
                            inplace=True)
        chunk_data.rio.write_crs('wgs84', inplace=True)
        chunk_data = chunk_data.assign_coords({
            'lon': ((chunk_data.lon- 180) % 360) - 180})
        synth_huc_ts_list = [find_huc_ts(chunk_data, geom) for geom in geom_subset]
        synth_huc_regions = xr.concat(synth_huc_ts_list, dim='region')
        synth_huc_regions = synth_huc_regions.assign_coords({'region': name_list})
        synth_huc_regions.to_netcdf(output_dir + f'synthLE_huc_ts_mem{ens_mem:02}_chunk{i}.nc')
    
