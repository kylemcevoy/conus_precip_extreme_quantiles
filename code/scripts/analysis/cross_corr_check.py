import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from scipy.signal import welch

def lagged_corr(data, mode1, mode2, shifts):
    Y = data.copy()
    ts1 = Y[mode1]
    ts2 = Y[mode2]
    lagged_corr_np = np.zeros((shifts.shape[0]))
    shift_index = np.arange(shifts.shape[0])
    for j in shift_index:
        lagged_corr = ts1.shift(periods=shifts[j]).corr(ts2)
        lagged_corr_np[j] = lagged_corr
    return lagged_corr_np

def lagged_corr_surrogate(surr_modes1, surr_modes2, shifts):
    time_indx = pd.date_range('1920-01-01', '2020-12-01', freq='MS')
    N = surr_modes1.shape[0]
    p = shifts.shape[0]
    lagged_corr_np = np.zeros((N, p))
    for n in np.arange(N):
        surr_ts1 = pd.Series(surr_modes1[n], index=time_indx)
        surr_ts2 = pd.Series(surr_modes2[n], index=time_indx)
        for j in np.arange(p):
                    lagged_corr_np[n, j] = (surr_ts1.
                        shift(periods=shifts[j]).
                        corr(surr_ts2))
    return lagged_corr_np


ortho_modes = xr.open_dataset('/home/data/projects/conus_precip_extremes/obsLE/gpcc_cvdp/ortho_modes.nc')
synth_modes = xr.open_mfdataset('/home/data/projects/conus_precip_extremes/obsLE/gpcc_cvdp/surrogate_modes.nc')

synth_modes1 = synth_modes.isel(ens_mem=1)
synth_modes1 = synth_modes1.compute()

synth_modes2 = synth_modes.sel(ens_member=543).load()

synth_modes1['enso']

fs = 12
plot_freq, enso_psd = welch(ortho_modes['enso'], fs=12)
_, pdo_psd = welch(ortho_modes['pdo'], fs=12)
_, pna_psd = welch(ortho_modes['pna'], fs=12)
_, nao_psd = welch(ortho_modes['nao'], fs=12)

def plot_spectra(mode, ortho_modes, synth_modes):
    fs = 12
    plot_freq, psd = welch(ortho_modes[mode], fs=fs)
    _, synth_psd = welch(synth_modes[mode], fs=fs)
    
    plt.plot(plot_freq, psd)
    plt.plot(plot_freq, synth_psd)
    plt.semilogx()
    plt.semilogy()

plot_spectra('pna', ortho_modes, synth_modes2)
plt.close()

plt.plot(plot_freq, enso_psd)
plt.semilogx()
plt.semilogy()
plt.close()

plt.plot(plot_freq, pdo_psd)
plt.semilogx()
plt.semilogy()
plt.close()

plt.plot(plot_freq, pna_psd)
plt.semilogx()
plt.semilogy()
plt.close()

plt.plot(plot_freq, nao_psd)
plt.semilogx()
plt.semilogy()
plt.close()

synth_modes_sub = synth_modes.sel(ens_mem=slice(0, 99))
synth_modes_sub = synth_modes_sub.compute()
shifts = np.arange(-300, 300)

ortho_modes_df = ortho_modes.to_dataframe()

enso_pdo_corr = lagged_corr(ortho_modes_df, 'enso', 'pdo', shifts)
synth_enso_pdo_corr = lagged_corr_surrogate(synth_modes_sub['enso'],
                                            synth_modes_sub['pdo'],
                                            shifts=shifts)

enso_pna_corr = lagged_corr(ortho_modes_df, 'enso', 'pna', shifts)
synth_enso_pna_corr = lagged_corr_surrogate(synth_modes_sub['enso'],
                                            synth_modes_sub['pna'],
                                            shifts=shifts)

pdo_pna_corr = lagged_corr(ortho_modes_df, 'pdo', 'pna', shifts)
synth_pdo_pna_corr = lagged_corr_surrogate(synth_modes_sub['pdo'],
                                            synth_modes_sub['pna'],
                                            shifts=shifts)


plt.plot(shifts, enso_pdo_corr)
for i in range(10):
    plt.plot(shifts, synth_enso_pdo_corr[i])

plt.close()

mpl.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(dpi=400, figsize=(10, 8))
ax.plot(shifts[300:330], synth_enso_pdo_corr[0, 300:330], color='black', alpha=0.4, label='surrogate modes')
for i in range(1, 10):
    ax.plot(shifts[300:330], synth_enso_pdo_corr[i, 300:330], color='black', alpha=0.4)
ax.plot(shifts[300:330], enso_pdo_corr[300:330], label='observational modes', color='red')
ax.set_xlabel('lag ')
ax.set_ylabel('correlation')
plt.legend()
ax.set_title('Correlation of PDO with Lagged ENSO')
#plt.savefig(plot_dir + )

plt.close()

plt.plot(shifts[301:330], enso_pna_corr[301:320])
for i in range(10):
    plt.plot(shifts[301:320], synth_enso_pna_corr[i, 301:320])

plt.close()

plt.plot(shifts, enso_pna_corr)
for i in range(10):
    plt.plot(shifts, synth_enso_pna_corr[i], alpha=0.3)

plt.close()

plt.plot(shifts[280:320], pdo_pna_corr[280:320])
for i in range(10):
    plt.plot(shifts[280:320], synth_pdo_pna_corr[i, 280:320], alpha=0.3)

plt.close()

plt.plot(shifts, pdo_pna_corr)
for i in range(10):
    plt.plot(shifts, synth_pdo_pna_corr[i], alpha=0.3)

ortho_modes_cesm = xr.open_dataset('/home/data/projects/conus_precip_extremes/synthLE/mem00/ortho_modes.nc')
synth_modes_cesm = xr.open_mfdataset('/home/data/projects/conus_precip_extremes/synthLE/mem00/surrogate_modes.nc')

synth_modes_cesm1 = synth_modes_cesm.sel(ens_member=0)
synth_modes_cesm1.load()

synth_modes_cesm2 = synth_modes_cesm.sel(ens_member=543).load()

synth_modes_cesm1['enso']

fs = 12
plot_freq, enso_psd = welch(ortho_modes_cesm['enso'], fs=12)
_, pdo_psd = welch(ortho_modes_cesm['pdo'], fs=12)
_, pna_psd = welch(ortho_modes_cesm['pna'], fs=12)
_, nao_psd = welch(ortho_modes_cesm['nao'], fs=12)

plot_spectra('nao', ortho_modes, ortho_modes_cesm)
plot_spectra('enso', ortho_modes_cesm, synth_modes_cesm1)
plt.close()

plt.plot(plot_freq, enso_psd)
plt.semilogx()
plt.semilogy()
plt.close()

plt.plot(plot_freq, pdo_psd)
plt.semilogx()
plt.semilogy()
plt.close()

plt.plot(plot_freq, pna_psd)
plt.semilogx()
plt.semilogy()
plt.close()

plt.plot(plot_freq, nao_psd)
plt.semilogx()
plt.semilogy()
plt.close()

synth_modes_cesm_sub = synth_modes_cesm.sel(ens_member=slice(0, 99))
synth_modes_cesm_sub.load()

shifts = np.arange(-300, 300)

ortho_modes_cesm_df = ortho_modes_cesm.to_dataframe()

enso_pdo_corr = lagged_corr(ortho_modes_cesm_df, 'enso', 'pdo', shifts)
synth_enso_pdo_corr = lagged_corr_surrogate(synth_modes_cesm_sub['enso'],
                                            synth_modes_cesm_sub['pdo'],
                                            shifts=shifts)

enso_pna_corr = lagged_corr(ortho_modes_cesm_df, 'enso', 'pna', shifts)
synth_enso_pna_corr = lagged_corr_surrogate(synth_modes_cesm_sub['enso'],
                                            synth_modes_cesm_sub['pna'],
                                            shifts=shifts)

pdo_pna_corr = lagged_corr(ortho_modes_cesm_df, 'pdo', 'pna', shifts)
synth_pdo_pna_corr = lagged_corr_surrogate(synth_modes_cesm_sub['pdo'],
                                            synth_modes_cesm_sub['pna'],
                                            shifts=shifts)



rng = np.random.default_rng(4743105)

### Output directory -- (end with trailing slash)
proj_dir = '/home/data/projects/conus_precip_extremes/'
save_dir = proj_dir + 'obsLE/gpcc_resid/'
mode_path = proj_dir + 'climate_modes/'

#using pre-processed gpcc otherwise comment this out and 
# uncomment the Load GPCC code
gpcc_path = proj_dir + 'gpcc/gpcc_mmday.nc'
gpcc_mmday = xr.open_dataarray(gpcc_path)   

### Climate Modes
mode_df = xr.open_dataset(mode_path + 'cvdp_obs_modes.nc')
mode_df = mode_df.to_dataframe()
# setting (model_mode_list = None) uses all modes: ENSO, PDO, PNA, NAO.
mode_list = ['enso', 'pdo', 'pna', 'nao']
# the modes that are calculated using multivariate iaaft
mv_mode_list = []
fit_seasonal = [True, False, True, True]
mv_fit_seasonal = []

ortho_modes = xr.open_dataset('/home/data/projects/conus_precip_extremes/obsLE/gpcc_cvdp/ortho_modes.nc')
ortho_mode_df = ortho_modes.to_pandas()
n_ens_members = 10

from obsLE import resample

surrogate_modes = resample.create_surrogate_modes(ortho_mode_df,
                                                  mode_list=mode_list,
                                                      fit_seasonal=fit_seasonal,
                                                      mv_mode_list=None,
                                                      mv_fit_seasonal=None,
                                                      n_ens_members=n_ens_members,
                                                      rng=rng,
                                                      save_dir='/home/data/projects/conus_precip_extremes/')

test_surrogates = xr.open_dataset('/home/data/projects/conus_precip_extremes/surrogate_modes.nc')

test_surrogates

enso_pdo_corr2 = lagged_corr(ortho_modes_df, 'enso', 'pdo', shifts)
synth_enso_pdo_corr2 = lagged_corr_surrogate(test_surrogates['enso'],
                                            test_surrogates['pdo'],
                                            shifts=shifts)



fig, ax = plt.subplots(dpi=400, figsize=(10, 8))
ax.plot(shifts[300:330], synth_enso_pdo_corr2[0, 300:330], color='black', alpha=0.4, label='surrogate modes')
for i in range(1, 10):
    ax.plot(shifts[300:330], synth_enso_pdo_corr2[i, 300:330], color='black', alpha=0.4)
ax.plot(shifts[300:330], enso_pdo_corr2[300:330], label='observational modes', color='red')
ax.set_xlabel('lag ')
ax.set_ylabel('correlation')
plt.legend()
ax.set_title('Correlation of PDO with Lagged ENSO')