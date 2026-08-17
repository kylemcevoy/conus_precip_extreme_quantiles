import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import welch
from numpy.fft import rfft
from numpy.fft import rfftfreq

from obsLE import process_data as data_proc
from obsLE import resample

### Climate Mode Parameters
mode_path = '/home/data/projects/conus_precip_extremes/climate_modes/'
start_year = '1920'
end_year = '2020'
# setting (model_mode_list = None) uses all modes: ENSO, PDO, PNA, NAO, AO.
mode_list = ['enso', 'pdo', 'pna', 'nao', 'ao']
fit_seasonal = [mode != 'pdo' for mode in mode_list]

ortho_mode_df = data_proc.build_ortho_mode_df(mode_df=None,
                                              start_year='1920',
                                              end_year='2020',
                                              mode_list=mode_list,
                                              save_path=None,
                                              mode_path=mode_path)

n_ens_members = 100
surrogate_modes = resample.create_surrogate_modes(ortho_mode_df,
                                                  fit_seasonal=fit_seasonal,
                                                  n_ens_members=n_ens_members,
                                                  save_path=None)

surrogate_modes.sel(ens_member=1)

surrogate_modes.shape

fig, ax = plt.subplots()
for i in range(5):
    ax.plot(surrogate_modes[i, 0])
plt.close()

fs = 12
freq, psd = welch(surrogate_modes[0:10, 0], fs=fs)
enso_freq, enso_psd = welch(ortho_mode_df['enso'], fs=fs)

enso_dft = rfft(ortho_mode_df['enso'], norm='ortho')

enso_power = np.abs(enso_dft)**2
enso_freq2 = rfftfreq(n=1212, d=(1/12))

test_surr_dft = rfft(surrogate_modes[0:10, 0], norm='ortho')
test_surr_power = np.abs(test_surr_dft)**2

plt.plot(enso_freq2, enso_power)
plt.close()

for i in range(10):
    plt.plot(enso_freq, test_surr_power[i] - enso_power)
plt.plot(enso_freq, np.mean(test_surr_power - enso_power, axis=0))
plt.close()

fig, ax = plt.subplots()
for i in range(10):
    ax.plot(freq, psd[i], alpha=0.20)
ax.plot(enso_freq, enso_psd)
ax.vlines(0.20, ymin=0, ymax=2.5, linestyle='dashed', color='red')
plt.close()

pdo_freq, pdo_psd = welch(ortho_mode_df['pdo'], fs=fs)
surr_pdo_freq, surr_pdo_psd = welch(surrogate_modes[80:90, 1], fs=fs)

fig, ax = plt.subplots()
ax.plot(pdo_freq, pdo_psd)
for i in range(10):
    ax.plot(surr_pdo_freq, surr_pdo_psd[i], alpha=0.25)
ax.vlines(0.1, ymin=0, ymax=2.3)
plt.close()

pna_freq, pna_psd = welch(ortho_mode_df['pna'], fs=fs)
surr_pna_freq, surr_pna_psd = welch(surrogate_modes[70:80, 2], fs=fs)

fig, ax = plt.subplots()
ax.plot(pna_freq, pna_psd)
for i in range(10):
    ax.plot(surr_pna_freq, surr_pna_psd[i], alpha=0.25)
plt.close()

nao_freq, nao_psd = welch(ortho_mode_df['nao'], fs=fs)
surr_nao_freq, surr_nao_psd = welch(surrogate_modes[70:80, 3], fs=fs)

fig, ax = plt.subplots()
ax.plot(nao_freq, nao_psd)
for i in range(10):
    ax.plot(surr_nao_freq, surr_nao_psd[i], alpha=0.25)
plt.close()

ao_freq, ao_psd = welch(ortho_mode_df['ao'], fs=fs)
surr_ao_freq, surr_ao_psd = welch(surrogate_modes[70:80, 4], fs=fs)

fig, ax = plt.subplots()
ax.plot(ao_freq, ao_psd)
for i in range(10):
    ax.plot(surr_ao_freq, surr_ao_psd[i], alpha=0.25)
plt.close()