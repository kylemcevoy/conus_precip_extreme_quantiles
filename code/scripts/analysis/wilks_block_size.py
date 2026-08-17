import numpy as np
import xarray as xr

from scipy.optimize import minimize

data_dir = '/home/data/projects/conus_precip_extremes/obsLE/gpcc_cvdp/'
cesm2_dir = '/home/data/projects/conus_precip_extremes/synthLE/cesm2/mem27/'

residuals = xr.open_dataset(data_dir + 'regression_output.nc')
residuals = residuals['residuals']

def wilks_eq(L, n, vif):
    return (n - L + 1)**((2/3) * (1 - (1 / vif)))

def wilks_diff(L, n, vif):
    return np.abs(L - wilks_eq(L, n, vif))

def find_L(n, vif):
    L_init = np.sqrt(n)
    minimizer = minimize(lambda x: wilks_diff(x, n, vif), L_init)
    return np.floor(minimizer['x'][0])

def find_wilks_L(month_data):
    n = month_data.shape[0]
    rho = np.corrcoef(month_data[1:], month_data[:-1])[0, 1]
    vif = (1 + rho) / (1 - rho)
    v_prime = vif * np.exp(2 * vif / n)
    
    return find_L(n, v_prime)

nan_mask = ~(residuals.isnull().any('time'))

residuals_values = residuals.values[:, nan_mask]
n_loc = residuals_values.shape[1]

month_ind = np.arange(1212) % 12 + 1
output = np.zeros((12, residuals_values.shape[1]))

for i in range(12):
    for j in range(n_loc):
        month_data = residuals_values[i::12, j]
        output[i, j] = find_wilks_L(month_data)

nan_array =  np.zeros_like(residuals.groupby('time.month').mean())      
nan_array.fill(np.nan)

nan_array[:, nan_mask] = output

wilks = xr.DataArray(data=nan_array, 
                     coords={'month':np.arange(1, 13),
                             'lat': residuals.lat,
                             'lon': residuals.lon})

wilks.sel(month=12).plot(levels=np.linspace(0, 4.5, 9))

wilks.max('month').plot(levels=np.linspace(0.99, 3.99, 4))

wilks.groupby(wilks.month).quantile(0.97, dim=['lat', 'lon'])
wilks.quantile(0.97)

cesm2_regression = xr.open_dataset(cesm2_dir + 'regression_output.nc')
cesm2_residuals = cesm2_regression['residuals']

nan_mask_cesm = ~(cesm2_residuals.isnull().any(['time']))

cesm2_values = cesm2_residuals.values[:, nan_mask_cesm]
n_loc_cesm = cesm2_values.shape[1]

output_cesm = np.zeros((12, cesm2_values.shape[1]))

for i in range(12):
    for j in range(n_loc_cesm):
        month_data = cesm2_values[i::12, j]
        output_cesm[i, j] = find_wilks_L(month_data)

cesm_nan_array =  np.zeros_like(cesm2_residuals.groupby('time.month').mean())      
cesm_nan_array.fill(np.nan)

cesm_nan_array[:, nan_mask_cesm] = output_cesm

wilks_cesm = xr.DataArray(data=cesm_nan_array, coords={
                                                       'month':np.arange(1, 13),
                                                       'lat': cesm2_residuals.lat,
                                                       'lon': cesm2_residuals.lon})

wilks_cesm.quantile(0.97)
