import regionmask
import xarray as xr

output_dir = '/home/data/projects/conus_precip_extremes/gpcc/'

start_year = '1920'
end_year = '2020'

### Load GPCC
gpcc_path = '/home/data/GPCC/monthly/*_10.nc'
##### lat/lon coord ranges
lat_max = 50
lat_min = 24.5
lon_min = -126
lon_max = -65

gpcc = xr.open_mfdataset(gpcc_path)
gpcc = gpcc['precip']
gpcc = gpcc.sel(time=slice(start_year, end_year))

# pay attention to slicing order
gpcc_na = gpcc.sel(lat=slice(lat_max, lat_min),
                   lon=slice(lon_min, lon_max))

countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
US_mask = countries.mask(gpcc_na.lon, gpcc_na.lat) == 4

gpcc_na = gpcc_na.where(US_mask)
gpcc_na = gpcc_na.compute().astype('float64')

gpcc_na = gpcc_na.rename('precip')
gpcc_na.to_netcdf(output_dir + 'gpcc_totals.nc')

days_in_month = gpcc_na.time.dt.days_in_month
gpcc_mmday = gpcc_na / days_in_month

gpcc_mmday.to_netcdf(output_dir + 'gpcc_mmday.nc')
