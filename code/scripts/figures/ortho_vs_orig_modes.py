import matplotlib as mpl
import xarray as xr

cvdp_dir = '/home/data/projects/conus_precip_extremes/cvdp/obs_data/'
obsLE_dir = '/home/data/projects/conus_precip_extremes/obsLE/gpcc_cvdp/'
proj_dir = '/home/data/projects/conus_precip_extremes/'
mode_path = proj_dir + 'climate_modes/'
plot_dir = proj_dir + 'plots/figures/'

ersst = xr.open_dataset(cvdp_dir + 'ersstv5.185401-202412.nc',
                          decode_times=False)

sst_anoms = ersst['ssta']
era_data = xr.open_dataset(cvdp_dir + 'era20c_era5.mon.mean.msl.190001-202412.nc')

ersst_time = xr.date_range('1854-01-01', '2024-12-01', freq='MS')
ersst = ersst.assign_coords({'time': ersst_time})
ersst.time

modes = xr.open_dataset(mode_path + 'cvdp_obs_modes.nc')

orthog_modes = xr.open_dataset(obsLE_dir + 'ortho_modes.nc')
ortho_modes_df = orthog_modes.to_pandas()

ersst_1920_2020 = ersst['ssta'].sel(time=slice('1920', '2020'))

def fit_linear_models(y, X):
    """Fit OLS models of each individual column (locations) of y regressed against
    the design matrix X. The models are fit separately for each month by 
    subsetting the rows of y and X. A total of 12 * #{locations} independent
    regression models are fit.

    Parameters
    ----------
    y: 2D numpy array
        Matrix containing n observation rows against L columns containing time
        series of output data. L is an index for the locations at which the
        time series are recorded.
    X: 2D numpy array
        Design matrix containing n observation rows against p variable columns
        (including intercept). The columns of X should contain time series
        of monthly data containing only complete years, so that n is
        divisible by 12. X should contain an intercept column.

    Returns
    -------
    tuple(beta, RSS, residuals, fitted_values)

    beta: numpy array of shape (12, p, L)
        Contains the fitted regression coefficients indexed by 
        (month, variable, location). For each regression model fit using 
        y[I{month_j}] ~ X[I{month_j}], where I{month_j} is an indicator for the
        rows coming from month j. If L = 1, the last dimension is dropped.

    RSS: numpy array of shape (12, L)
        Contains the residual sums of squares of each fitted model. If L = 1, the
        last dimension is dropped.
        
    residuals: numpy array of shape (12, n // 12, L)
        Contains the residuals of each fitted model split by month. If L = 1, the
        last dimension is dropped.

    fitted_values: numpy array of shape (12, n // 12, l)
        Contains the fitted values of each regression model split by month. If
        L = 1 the last dimension is dropped.
    """
    
    n = X.shape[0]
    # referred to as p + 1 in documentation
    p = X.shape[1]
    
    if y.ndim > 1:
        L = y.shape[1]
        
        beta = np.zeros((12, p, L))
        residuals = np.zeros((12, n // 12, L))
        fitted_values = np.zeros((12, n // 12, L))
        RSS = np.zeros((12, L))
    else:
        beta = np.zeros((12, p))
        residuals = np.zeros((12, n // 12))
        fitted_values = np.zeros((12, n // 12))
        RSS = np.zeros((12))
        
    I_month = [(np.arange(n) % 12) == i for i in np.arange(12)]

    for i in range(12):
        beta[i, ...], RSS[i], *_ = np.linalg.lstsq(X[I_month[i]],
                                                   y[I_month[i]],
                                                   rcond=None)
        fitted_values[i] = (X[I_month[i]] @ beta[i, ...])
        residuals[i] = y[I_month[i]] - (X[I_month[i]] @ beta[i, ...])

    return beta, RSS, residuals, fitted_values

def build_xr(beta_array, nan_mask, var_names, lat_coord, lon_coord):
    p = beta_array.shape[0]
    lat_len = lat_coord.shape[0]
    lon_len = lon_coord.shape[0]
    
    beta_xr_data = np.empty((p, 12, lat_len, lon_len))
    beta_xr_data.fill(np.nan)
    beta_xr_data[:, :, nan_mask] = beta_array

    beta_ds = xr.Dataset(coords={'month': np.arange(1, 13),
                             'lat': lat_coord,
                             'lon': lon_coord})

    for i, var in enumerate(var_names):
        beta_ds[var] = (('month', 'lat', 'lon'),
                        beta_xr_data[i])
    
    return beta_ds


import pandas as pd 
pdo_design = ortho_modes_df[['intercept', 'pdo']]
pdo_orig = pd.DataFrame({'intercept': ortho_modes_df['intercept'], 'pdo': modes['pdo']})

pna_design = ortho_modes_df[['intercept', 'pna']]
nao_design = ortho_modes_df[['intercept', 'nao']]

ersst_nan_mask = ~ersst_1920_2020.isnull().any('time')
ersst_std = ersst_1920_2020.groupby('time.month') / ersst_1920_2020.groupby('time.month').std('time', ddof=1)
ersst_finite = np.isfinite(ersst_std).all('time')
ersst_flat = ersst_std.values[..., ersst_finite]
beta_orthog, *_ = fit_linear_models(ersst_flat, pdo_design.values)
beta_orig, *_ = fit_linear_models(ersst_flat, pdo_orig.values)

betas = build_xr(beta_out.transpose((1, 0, 2)), ersst_finite, ['intercept', 'pdo'], ersst_std.lat, ersst_std.lon)
betas_orig = build_xr(beta_orig.transpose((1, 0, 2)), ersst_finite, ['intercept', 'pdo'], ersst_std.lat, ersst_std.lon)


betas['pdo'].sel(month=12).plot(ax=ax[0], levels=np.linspace(-0.8, 0.8, 17), add_colorbar=False)
betas_orig['pdo'].sel(month=12).plot(ax=ax[1], levels=np.linspace(-0.8, 0.8, 17), add_colorbar=False)

beta_out

cmap_big = mpl.cm.RdBu_r
levels_big = np.linspace(-0.8, 0.8, 17)
norm_big = mpl.colors.BoundaryNorm(levels_big, cmap_big.N, extend='both')

fig, ax = plt.subplots(ncols=2,
                       constrained_layout=True, 
                       subplot_kw = {'projection': ccrs.PlateCarree(central_longitude=180)})
betas_orig['pdo'].sel(month=12).plot(ax=ax[0],
                                     transform=ccrs.PlateCarree(),
                                     levels=np.linspace(-0.8, 0.8, 17),
                                     add_colorbar=False)
betas['pdo'].sel(month=12).plot(ax=ax[1],
                                transform=ccrs.PlateCarree(),
                                levels=np.linspace(-0.8, 0.8, 17), 
                                add_colorbar=False)
ax[0].set_title('Original PDO')
ax[1].set_title('Orthogonalized PDO')
ax[0].coastlines()
ax[1].coastlines()
cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap_big, norm=norm_big),
                    pad=0.05,
                    ax=ax,
                    shrink=0.6,
                    orientation='horizontal')
cbar.set_label('Regression Coefficients []', fontsize=12)
fig.savefig(plot_dir + 'orig_vs_ortho_pdo_dec.png')


