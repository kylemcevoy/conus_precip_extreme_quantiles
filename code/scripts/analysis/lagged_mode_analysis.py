import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import ecdf
from scipy.signal import welch
from statsmodels.graphics.tsaplots import plot_acf

from obsLE import process_data as data_proc
from obsLE import resample

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

mode_path = '/home/data/projects/conus_precip_extremes/climate_modes/'
start_year = '1920'
end_year = '2020'
# setting (model_mode_list = None) uses all modes: ENSO, PDO, PNA, NAO, AO.
mode_list = ['enso', 'pdo', 'pna', 'nao', 'ao']
fit_seasonal = [mode != 'pdo' for mode in mode_list]

mode_df = data_proc.process_climate_modes(mode_path=mode_path)

ortho_mode_df = data_proc.build_ortho_mode_df(mode_df=None,
                                              start_year='1920',
                                              end_year='2020',
                                              mode_list=mode_list,
                                              save_path=None,
                                              mode_path=mode_path)

enso = ortho_mode_df['enso']
pdo = ortho_mode_df['pdo']
pna = ortho_mode_df['pna']

enso.shift(-3).loc['1920-01-01'] == enso.loc['1920-04-01']
enso.shift(3).loc['1920-07-01'] == enso.loc['1920-04-01']

enso.shift(3)[0:12]
enso.shift(9).corr(pdo)

enso.shift(9)[0:12]

enso_np = enso.values
pdo_np = pdo.values
pna_np = pna.values

bins = np.linspace(-4, 4, 25)

plt.hist(pdo_np, bins=bins, alpha=0.5, density=True)
plt.hist(pdo_np[enso_np < -1], bins=bins, alpha=0.5, density=True)
plt.hist(pdo_np[enso_np > 1], bins=bins, alpha=0.5, density=True)
plt.close()

deseason_modes = (mode_df 
  - mode_df.groupby(mode_df.index.month)
  .transform('mean')
  )

deseason_modes.groupby(deseason_modes.index.month).mean()

deseason_modes.std(ddof=1)

deseason_modes.groupby(deseason_modes.index.month).std(ddof=1)
deseason_modes.std(ddof=1)

deseason_modes = deseason_modes / deseason_modes.groupby(deseason_modes.index.month).transform('std', 1)

mode_df.mean()
deseason_modes.mean()

deseason_modes.std(ddof=1)

deseason_modes.corr()

deseason_ortho_modes = data_proc.orthogonalize_modes(deseason_modes)

deseason_ortho_modes.std(ddof=1)

deseason_ortho_modes.groupby(deseason_ortho_modes.index.month).std()

enso_pdo_corrs_ds = lagged_corr(data=deseason_ortho_modes,
  mode1='enso',
  mode2='pdo',
  shifts=np.arange(-300, 300))

enso_pdo_corrs = lagged_corr(data=ortho_mode_df,
  mode1='enso',
  mode2='pdo',
  shifts=np.arange(-300, 300))

enso_pdo_corrs_ds = lagged_corr(data=deseason_ortho_modes,
  mode1='enso',
  mode2='pdo',
  shifts=np.arange(-300, 300))

enso_pdo_corrs = lagged_corr(data=ortho_mode_df,
  mode1='enso',
  mode2='pdo',
  shifts=np.arange(-300, 300))

pdo_pna_corrs_ds = lagged_corr(data=deseason_ortho_modes,
  mode1='pdo',
  mode2='pna',
  shifts=np.arange(-300, 300))

pdo_pna_corrs = lagged_corr(data=ortho_mode_df,
  mode1='pdo',
  mode2='pna',
  shifts=np.arange(-300, 300))

shifts = np.arange(-300, 300)

plt.plot(shifts, enso_pdo_corrs, label='ortho modes')
plt.plot(shifts, enso_pdo_corrs_ds, label='ortho modes deseasoned')
plt.legend()
plt.xlabel('lag [+ goes back in time]')
plt.ylabel('correlation')
plt.title('pdo corr. with lagged enso (orthogonalized)')
plt.savefig('/home/data/projects/conus_precip_extremes/plots/lagged_enso_pdo_corrs_deseason.png')
plt.close()

shifts[np.argmax(enso_pdo_corrs)]
enso_pdo_corrs[301:324]

plt.plot(shifts, pdo_pna_corrs, label='ortho modes')
plt.plot(shifts, pdo_pna_corrs_ds, label='ortho modes deseasoned')
plt.legend()
plt.xlabel('lag [+ goes back in time]')
plt.ylabel('correlation')
plt.title('pna corr. with lagged pdo (orthogonalized)')
plt.savefig('/home/data/projects/conus_precip_extremes/plots/lagged_pdo_pna_corrs_deseason.png')
plt.close()

n_ens_members = 100
surrogate_modes = resample.create_surrogate_modes(ortho_mode_df,
                                                  fit_seasonal=fit_seasonal,
                                                  n_ens_members=n_ens_members,
                                                  save_path=None)

surrogate_enso_pdo_corrs = lagged_corr_surrogate(surr_modes1=surrogate_modes['enso'].values,
                                                surr_modes2=surrogate_modes['pdo'].values, 
                                                shifts=np.arange(-300, 300))

surr_pdo_pna_corrs = lagged_corr_surrogate(surr_modes1=surrogate_modes['pdo'].values,
                                           surr_modes2=surrogate_modes['pna'].values, 
                                           shifts=np.arange(-300, 300))

ep_lower_quant_pw, ep_upper_quant_pw = np.quantile(surrogate_enso_pdo_corrs,
                                        q=[0.025, 0.975], 
                                        axis=0)

pp_lower_quant_pw, pp_upper_quant_pw = np.quantile(surr_pdo_pna_corrs,
                                        q=[0.025, 0.975], 
                                        axis=0)

fig, ax = plt.subplots(dpi=200)
for i in range(surrogate_enso_pdo_corrs.shape[0]):
    ax.plot(np.arange(-300, 300),
     surrogate_enso_pdo_corrs[i],
      color='k',
      alpha=0.2)
ax.plot(np.arange(-300, 300), enso_pdo_corrs, color='red')
ax.plot(np.arange(-300, 300), ep_lower_quant_pw, color='blue', linestyle='--')
ax.plot(np.arange(-300, 300), ep_upper_quant_pw, color='blue', linestyle='--')
ax.set_title('lagged enso correlations with pdo (surrogate correlations in black)')
ax.set_xlabel('lags')
ax.set_ylabel('correlation')
fig.savefig('/home/data/projects/conus_precip_extremes/plots/mode_coherence/lagged_enso_pdo_corr_with_surr.png')
plt.close()

fig, ax = plt.subplots(dpi=200)
for i in range(surr_pdo_pna_corrs.shape[0]):
    ax.plot(np.arange(-300, 300),
     surr_pdo_pna_corrs[i],
      color='k',
      alpha=0.2)
ax.plot(np.arange(-300, 300), pdo_pna_corrs, color='red')
ax.plot(np.arange(-300, 300), pp_lower_quant_pw, color='blue', linestyle='--')
ax.plot(np.arange(-300, 300), pp_upper_quant_pw, color='blue', linestyle='--')
ax.set_title('lagged pdo correlations with pna (surrogate correlations in black)')
ax.set_xlabel('lags')
ax.set_ylabel('correlation')
fig.savefig('/home/data/projects/conus_precip_extremes/plots/mode_coherence/lagged_pdo_pna_corr_with_surr.png')
plt.close()

pdo_pna_corrs[298:306]

plt.plot(shifts[296:312], pdo_pna_corrs[296:312])
plt.title('lagged pdo corr with pna')
plt.savefig('/home/data/projects/conus_precip_extremes/plots/zoom_lagged_pdo_corr_pna.png')

pdo_np = ortho_mode_df['pdo'].values
enso_np = ortho_mode_df['enso'].values
pna_np = ortho_mode_df['pna'].values

pdo_ecdf = ecdf(pdo_np)
pdo_ecdf_neg_enso = ecdf(pdo_np[enso_np < -1])
pdo_ecdf_pos_enso = ecdf(pdo_np[enso_np > 1])

x_seq = np.linspace(-3, 3)

fig, ax = plt.subplots()
pdo_ecdf.cdf.plot(ax=ax, label='full dist.')
pdo_ecdf_neg_enso.cdf.plot(ax=ax, label='enso < -1')
pdo_ecdf_pos_enso.cdf.plot(ax=ax, label='enso > 1')
plt.legend()
plt.title('orthog. pdo ECDF conditioned on enso values')
fig.savefig('/home/data/projects/conus_precip_extremes/plots/orthog_pdo_ecdf_enso_cond.png')
plt.close()

def lag_arrays(arr1, arr2, lag):
  if lag >= 0:
    arr1 = arr1[lag:]
    arr2 = arr2[:-lag]
  else:
    arr1 = arr1[:lag]
    arr2 = arr2[-lag:]
  return arr1, arr2

enso_lag3, pdo_j3 = lag_arrays(enso_np, pdo_np, 3)
enso_lag6, pdo_j6 = lag_arrays(enso_np, pdo_np, 6)
enso_lag9, pdo_j9 = lag_arrays(enso_np, pdo_np, 9)
enso_lag12, pdo_j12 = lag_arrays(enso_np, pdo_np, 12)

pdo_lag3, pna_j3 = lag_arrays(pdo_np, pna_np, 3)
pdo_lag6, pna_j6 = lag_arrays(pdo_np, pna_np, 6)
pdo_lag9, pna_j9 = lag_arrays(pdo_np, pna_np, 9)
pdo_lag12, pna_j12 = lag_arrays(pdo_np, pna_np, 12)

fig, ax = plt.subplots()
ecdf(pdo_j3).cdf.plot(label='full dist', ax=ax)
ecdf(pdo_j3[enso_lag3 < -1]).cdf.plot(ax=ax, label='lag 3 enso < -1')
ecdf(pdo_j3[enso_lag3 > 1]).cdf.plot(ax=ax, label='lag 3 enso > 1')
ax.legend()
ax.set_title('orthog. pdo ecdf conditioned on lag 3 enso')
fig.savefig('/home/data/projects/conus_precip_extremes/plots/orthog_pdo_ecdf_lag3_enso_cond.png')
plt.close(fig)

fig, ax = plt.subplots()
ecdf(pdo_j6).cdf.plot(label='full dist', ax=ax)
ecdf(pdo_j6[enso_lag6 < -1]).cdf.plot(ax=ax, label='lag 6 enso < -1')
ecdf(pdo_j6[enso_lag6 > 1]).cdf.plot(ax=ax, label='lag 6 enso > 1')
ax.legend()
ax.set_title('orthog. pdo ecdf conditioned on lag 6 enso')
fig.savefig('/home/data/projects/conus_precip_extremes/plots/orthog_pdo_ecdf_lag6_enso_cond.png')
plt.close(fig)

fig, ax = plt.subplots()
ecdf(pdo_j9).cdf.plot(label='full dist', ax=ax)
ecdf(pdo_j9[enso_lag9 < -1]).cdf.plot(ax=ax, label='lag 9 enso < -1')
ecdf(pdo_j9[enso_lag9 > 1]).cdf.plot(ax=ax, label='lag 9 enso > 1')
ax.legend()
ax.set_title('orthog. pdo ecdf conditioned on lag 9 enso')
fig.savefig('/home/data/projects/conus_precip_extremes/plots/orthog_pdo_ecdf_lag9_enso_cond.png')
plt.close(fig)

fig, ax = plt.subplots()
ecdf(pdo_j12).cdf.plot(label='full dist', ax=ax)
ecdf(pdo_j12[enso_lag12 < -1]).cdf.plot(ax=ax, label='lag 12 enso < -1')
ecdf(pdo_j12[enso_lag12 > 1]).cdf.plot(ax=ax, label='lag 12 enso > 1')
ax.legend()
ax.set_title('orthog. pdo ecdf conditioned on lag 12 enso')
fig.savefig('/home/data/projects/conus_precip_extremes/plots/orthog_pdo_ecdf_lag12_enso_cond.png')
plt.close(fig)

plt.hist(pdo_np, 
  density=True, 
  bins=np.linspace(-4, 4, 19), 
  alpha=0.5,
  label='full dist')
plt.hist(pdo_np[enso_np < -1], 
  density=True, 
  bins=np.linspace(-4, 4, 19), 
  alpha=0.5,
  label='cond. enso < -1')
plt.hist(pdo_np[enso_np > 1],
  density=True,
  bins=np.linspace(-4, 4, 19),
  alpha=0.5,
  label='cond. enso > 1')
plt.legend()
plt.title('pdo conditioned on lag 0 enso')
plt.savefig('/home/data/projects/conus_precip_extremes/plots/pdo_hist_lag0_enso_cond.png')
plt.close()

plt.hist(pdo_j3, 
  density=True, 
  bins=np.linspace(-4, 4, 19), 
  alpha=0.5,
  label='full dist')
plt.hist(pdo_j3[enso_lag3 < -1], 
  density=True, 
  bins=np.linspace(-4, 4, 19), 
  alpha=0.5,
  label='cond. enso < -1')
plt.hist(pdo_j3[enso_lag3 > 1],
  density=True,
  bins=np.linspace(-4, 4, 19),
  alpha=0.5,
  label='cond. enso > 1')
plt.title('pdo conditioned on lag 3 enso')
plt.legend()
plt.savefig('/home/data/projects/conus_precip_extremes/plots/pdo_hist_lag3_enso_cond.png')
plt.close()

plt.hist(pdo_j6, 
  density=True, 
  bins=np.linspace(-4, 4, 19), 
  alpha=0.5,
  label='full dist')
plt.hist(pdo_j6[enso_lag6 < -1], 
  density=True, 
  bins=np.linspace(-4, 4, 19), 
  alpha=0.5,
  label='cond. enso < -1')
plt.hist(pdo_j6[enso_lag6 > 1],
  density=True,
  bins=np.linspace(-4, 4, 19),
  alpha=0.5,
  label='cond. enso > 1')
plt.title('pdo conditioned on lag 6 enso')
plt.savefig('/home/data/projects/conus_precip_extremes/plots/pdo_hist_lag6_enso_cond.png')
plt.close()

plt.hist(pdo_j9, 
  density=True, 
  bins=np.linspace(-4, 4, 19), 
  alpha=0.5,
  label='full dist')
plt.hist(pdo_j9[enso_lag9 < -1], 
  density=True, 
  bins=np.linspace(-4, 4, 19), 
  alpha=0.5,
  label='cond. enso < -1')
plt.hist(pdo_j9[enso_lag9 > 1],
  density=True,
  bins=np.linspace(-4, 4, 19),
  alpha=0.5,
  label='cond. enso > 1')
plt.title('pdo conditioned on lag 9 enso')
plt.savefig('/home/data/projects/conus_precip_extremes/plots/pdo_hist_lag9_enso_cond.png')
plt.close()

#### PNA cond on PDO

plt.hist(pna_np, 
  density=True, 
  bins=np.linspace(-4.5, 4.5, 21), 
  alpha=0.5,
  label='full dist')
plt.hist(pna_np[pdo_np < -1], 
  density=True, 
  bins=np.linspace(-4.5, 4.5, 21), 
  alpha=0.5,
  label='cond. pdo < -1')
plt.hist(pna_np[pdo_np > 1],
  density=True,
  bins=np.linspace(-4.5, 4.5, 21),
  alpha=0.5,
  label='cond. pdo > 1')
plt.legend()
plt.title('pna conditioned on lag 0 pdo')
plt.savefig('/home/data/projects/conus_precip_extremes/plots/pna_hist_lag0_pdo_cond.png')
plt.close()

plt.hist(pna_j3, 
  density=True, 
  bins=np.linspace(-4.5, 4.5, 21), 
  alpha=0.5,
  label='full dist')
plt.hist(pna_j3[pdo_lag3 < -1], 
  density=True, 
  bins=np.linspace(-4.5, 4.5, 21), 
  alpha=0.5,
  label='cond. pdo < -1')
plt.hist(pna_j3[pdo_lag3 > 1],
  density=True,
  bins=np.linspace(-4.5, 4.5, 21),
  alpha=0.5,
  label='cond. pdo > 1')
plt.title('pna conditioned on lag 3 pdo')
plt.legend()
plt.savefig('/home/data/projects/conus_precip_extremes/plots/pna_hist_lag3_pdo_cond.png')
plt.close()

plt.hist(pna_j6, 
  density=True, 
  bins=np.linspace(-4.5, 4.5, 21), 
  alpha=0.5,
  label='full dist')
plt.hist(pna_j6[pdo_lag6 < -1], 
  density=True, 
  bins=np.linspace(-4.5, 4.5, 21), 
  alpha=0.5,
  label='cond. pdo < -1')
plt.hist(pna_j6[pdo_lag6 > 1],
  density=True,
  bins=np.linspace(-4.5, 4.5, 21),
  alpha=0.5,
  label='cond. pdo > 1')
plt.title('pna conditioned on lag 6 pdo')
plt.savefig('/home/data/projects/conus_precip_extremes/plots/pna_hist_lag6_pdo_cond.png')
plt.close()

plt.hist(pna_j9, 
  density=True, 
  bins=np.linspace(-4.5, 4.5, 21), 
  alpha=0.5,
  label='full dist')
plt.hist(pna_j9[pdo_lag9 < -1], 
  density=True, 
  bins=np.linspace(-4.5, 4.5, 21), 
  alpha=0.5,
  label='cond. pdo < -1')
plt.hist(pna_j9[pdo_lag9 > 1],
  density=True,
  bins=np.linspace(-4.5, 4.5, 21),
  alpha=0.5,
  label='cond. pdo > 1')
plt.title('pna conditioned on lag 9 pdo')
plt.savefig('/home/data/projects/conus_precip_extremes/plots/pna_hist_lag9_pdo_cond.png')
plt.close()


fig, ax = plt.subplots()
ecdf(pna_np).cdf.plot(label='full dist', ax=ax)
ecdf(pna_np[pdo_np < -1]).cdf.plot(ax=ax, label='pdo < -1')
ecdf(pna_np[pdo_np > 1]).cdf.plot(ax=ax, label='pdo > 1')
ax.legend()
ax.set_title('orthog. pna ecdf conditioned on pdo')
fig.savefig('/home/data/projects/conus_precip_extremes/plots/orthog_pna_ecdf_lag0_pdo_cond.png')
plt.close(fig)

fig, ax = plt.subplots()
ecdf(pna_j3).cdf.plot(label='full dist', ax=ax)
ecdf(pna_j3[pdo_lag3 < -1]).cdf.plot(ax=ax, label='lag 3 pdo < -1')
ecdf(pna_j3[pdo_lag3 > 1]).cdf.plot(ax=ax, label='lag 3 pdo > 1')
ax.legend()
ax.set_title('orthog. pna ecdf conditioned on lag 3 pdo')
fig.savefig('/home/data/projects/conus_precip_extremes/plots/orthog_pna_ecdf_lag3_pdo_cond.png')
plt.close(fig)

fig, ax = plt.subplots()
ecdf(pna_j6).cdf.plot(label='full dist', ax=ax)
ecdf(pna_j6[pdo_lag6 < -1]).cdf.plot(ax=ax, label='lag 6 pdo < -1')
ecdf(pna_j6[pdo_lag6 > 1]).cdf.plot(ax=ax, label='lag 6 pdo > 1')
ax.legend()
ax.set_title('orthog. pna ecdf conditioned on lag 6 pdo')
fig.savefig('/home/data/projects/conus_precip_extremes/plots/orthog_pna_ecdf_lag6_pdo_cond.png')
plt.close(fig)

fig, ax = plt.subplots()
ecdf(pna_j9).cdf.plot(label='full dist', ax=ax)
ecdf(pna_j9[pdo_lag9 < -1]).cdf.plot(ax=ax, label='lag 9 pdo < -1')
ecdf(pna_j9[pdo_lag9 > 1]).cdf.plot(ax=ax, label='lag 9 pdo > 1')
ax.legend()
ax.set_title('orthog. pna ecdf conditioned on lag 9 pdo')
fig.savefig('/home/data/projects/conus_precip_extremes/plots/orthog_pna_ecdf_lag9_pdo_cond.png')
plt.close(fig)

surrogate_enso = surrogate_modes['enso']
surrogate_pdo = surrogate_modes['pdo']

surr_enso_ts1 = surrogate_enso[0]
surr_pdo_ts1 = surrogate_pdo[0]

surr_pdo1_ecdf = ecdf(surrogate_pdo[0])
enso_plus_pdo1s_ecdf = ecdf(surr_pdo_ts1[surr_enso_ts1 > 1])
enso_neg_pdo1s_ecdf = ecdf(surr_pdo_ts1[surr_enso_ts1 < -1])

surr_pdo1_ecdf.cdf.plot()
enso_plus_pdo1s_ecdf.cdf.plot()
enso_neg_pdo1s_ecdf.cdf.plot()
plt.close()

surr_enso_ts10 = surrogate_enso[10]
surr_pdo_ts10 = surrogate_pdo[10]

surr_pdo10_ecdf = ecdf(surrogate_pdo[10])
enso_plus_pdo10s_ecdf = ecdf(surr_pdo_ts10[surr_enso_ts10 > 1])
enso_neg_pdo10s_ecdf = ecdf(surr_pdo_ts10[surr_enso_ts10 < -1])

surr_pdo10_ecdf.cdf.plot()
enso_plus_pdo10s_ecdf.cdf.plot()
enso_neg_pdo10s_ecdf.cdf.plot()
plt.close()

fig, ax = plt.subplots()
plot_acf(mode_df['pdo'], ax=ax)
plot_acf(ortho_mode_df['pdo'], ax=ax)
plt.close()

fig, ax = plt.subplots()
plot_acf(mode_df['nao'], ax=ax)
plot_acf(ortho_mode_df['nao'], ax=ax)
plt.close()

fig, ax = plt.subplots()
plot_acf(mode_df['enso'], ax=ax)
plot_acf(mode_df['pdo'], ax=ax)
plt.close()

plt.hist2d(mode_df['enso'], mode_df['pdo'])
plt.close()

plt.hist2d(ortho_mode_df['enso'], ortho_mode_df['pdo'])
plt.close()

plt.hist2d(ortho_mode_df['enso'].shift(6).iloc[9:], ortho_mode_df['pdo'].iloc[9:])

np.corrcoef(pdo_j12, enso_lag12)

pdo_nonortho = mode_df['pdo']
enso_nonortho = mode_df['enso']
pna_nonortho = mode_df['pna']

nonortho_pdo_lagenso = lagged_corr(mode_df, 'enso', 'pdo', shifts=shifts)
nonortho_pna_lagpdo = lagged_corr(mode_df, 'pdo', 'pna', shifts=shifts)

nonortho_pdo_lagenso[300:312]

fig, ax = plt.subplots()
ax.plot(shifts[300:336], nonortho_pdo_lagenso[300:336], label='not orthog.', zorder=200)
ax.plot(shifts[300:336], enso_pdo_corrs[300:336], label='orthog pdo', zorder=250)
ax.plot(shifts[300:336], 
  surrogate_enso_pdo_corrs[i, 300:336],
  label='surrogate corrs',
  color='k',
  alpha=0.3)
for i in range(1, 100):
  ax.plot(shifts[300:336], surrogate_enso_pdo_corrs[i, 300:336], 
  color='k', 
  alpha=0.2)
ax.legend()
ax.set_title('orthog. pdo with lagged enso [0-36 months] 100 surrogates')
fig.savefig('/home/data/projects/conus_precip_extremes/plots/pdo_corr_enso_lag0-36_100surr.png')
plt.close()

fig, ax = plt.subplots()
ax.plot(shifts[300:336], nonortho_pdo_lagenso[300:336], label='not orthog.', zorder=200)
ax.plot(shifts[300:336], enso_pdo_corrs[300:336], label='orthog pdo', zorder=250)
ax.plot(shifts[300:336], 
  surrogate_enso_pdo_corrs[i, 300:336],
  label='surrogate corrs',
  color='k',
  alpha=0.3)
for i in range(1, 11):
  ax.plot(shifts[300:336], surrogate_enso_pdo_corrs[i, 300:336], 
  color='k', 
  alpha=0.2)
ax.legend()
ax.set_title('pdo corr. with lagged enso [0-36 months], 10 surrogates')
fig.savefig('/home/data/projects/conus_precip_extremes/plots/enso_pdo_corrs_lag0-36_plot.png')
plt.close()

fig, ax = plt.subplots()
ax.plot(shifts[300:336], nonortho_pna_lagpdo[300:336], label='not orthog.', zorder=200)
ax.plot(shifts[300:336], pdo_pna_corrs[300:336], label='orthog pdo/pna', zorder=250)
ax.plot(shifts[300:336], 
  surr_pdo_pna_corrs[i, 300:336],
  label='surrogate corrs',
  color='k',
  alpha=0.3)
for i in range(1, 11):
  ax.plot(shifts[300:336], surr_pdo_pna_corrs[i, 300:336], 
  color='k', 
  alpha=0.2)
ax.legend()
ax.set_title('orthog. pna corr. with lagged pdo [0-36 months], 10 surrogates')
fig.savefig('/home/data/projects/conus_precip_extremes/plots/pdo_pna_corrs_lag0-36_plot.png')
plt.close()

fig, ax = plt.subplots()
ax.plot(shifts[300:336], nonortho_pna_lagpdo[300:336], label='not orthog.', zorder=200)
ax.plot(shifts[300:336], pdo_pna_corrs[300:336], label='orthog pdo/pna', zorder=250)
ax.plot(shifts[300:336], 
  surr_pdo_pna_corrs[i, 300:336],
  label='surrogate corrs',
  color='k',
  alpha=0.3)
for i in range(1, 100):
  ax.plot(shifts[300:336], surr_pdo_pna_corrs[i, 300:336], 
  color='k', 
  alpha=0.2)
ax.legend()
ax.set_title('orthog. pna corr. with lagged pdo [0-36 months], 100 surrogates')
fig.savefig('/home/data/projects/conus_precip_extremes/plots/mode_coherence/pdo_pna_corrs_lag0-36_plot_100surr.png')
plt.close()



np.min(pdo_np)
np.max(pdo_np)



#### Older Code

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

surrogate_enso_pdo_corrs = lagged_corr_surrogate(surr_modes1=surrogate_modes[:, 0],
                                                surr_modes2=surrogate_modes[:, 1], 
                                                shifts=np.arange(-300, 300))


enso_pdo_corrs = lagged_corr(data=ortho_mode_df,
  mode1='enso',
  mode2='pdo',
  shifts=np.arange(-300, 300))


def lagged_corr_monthly(data, mode1, mode2, shifts):
    Y = data.copy()
    ts1 = Y[mode1]
    ts2 = Y[mode2]
    lagged_corr_np = np.zeros((12, shifts.shape[0]))
    shift_index = np.arange(shifts.shape[0])
    for i in np.arange(1, 13):
        for j in shift_index:
            ts1_month = ts1.loc[ts1.index.month == i]
            ts2_month = ts2.loc[ts2.index.month == i]
            lagged_corr = ts1_month.shift(periods=shifts[j]).corr(ts2_month)
            lagged_corr_np[i - 1, j] = lagged_corr
    return lagged_corr_np

enso_pdo_monthly_corrs = lagged_corr_monthly(data=ortho_mode_df,
mode1='enso',
mode2='pdo',
shifts=np.arange(-50, 50))

np.argmax(enso_pdo_corrs)

fig, ax = plt.subplots(dpi=200)
ax.plot(np.arange(-300, 300), enso_pdo_corrs)
ax.set_title('pdo correlation with lagged enso')
ax.set_xlabel('lags')
ax.set_ylabel('correlation')
fig.savefig('/home/data/projects/conus_precip_extremes/plots/mode_coherence/lagged_enso_pdo.png')
plt.close()

lower_quant_pw, upper_quant_pw = np.quantile(surrogate_enso_pdo_corrs,
                                        q=[0.025, 0.975], 
                                        axis=0)
lower_quant_over, upper_quant_over = np.quantile(surrogate_enso_pdo_corrs,
    q=[0.025, 0.975])

fig, ax = plt.subplots(dpi=200)
for i in range(surrogate_enso_pdo_corrs.shape[0]):
    ax.plot(np.arange(-300, 300),
     surrogate_enso_pdo_corrs[i],
      color='k',
      alpha=0.2)
ax.plot(np.arange(-300, 300), enso_pdo_corrs, color='red')
ax.plot(np.arange(-300, 300), lower_quant_pw, color='blue', linestyle='--')
ax.plot(np.arange(-300, 300), upper_quant_pw, color='blue', linestyle='--')
ax.set_title('lagged enso correlations with pdo (surrogate correlations in black)')
ax.set_xlabel('lags')
ax.set_ylabel('correlation')
fig.savefig('/home/data/projects/conus_precip_extremes/plots/mode_coherence/lagged_enso_pdo_corr_with_surr.png')
plt.close()

enso_pna_corrs = lagged_corr(data=ortho_mode_df,
  mode1='enso',
  mode2='pna',
  shifts=np.arange(-300, 300))

fig, ax = plt.subplots(dpi=200)
ax.plot(np.arange(-300, 300), enso_pna_corrs, color='orange')
ax.set_title('pna correlation with lagged enso')
ax.set_xlabel('lags')
ax.set_ylabel('correlation')
fig.savefig('/home/data/projects/conus_precip_extremes/plots/mode_coherence/lagged_enso_pna.png')
plt.close()

def lagged_mode_plots(corr_np, lagged_mode, mode2, plot_dir):
    fig, ax = plt.subplots(dpi=200)
    ax.plot(np.arange(-300, 300), corr_np)
    ax.set_title(f'{mode2} correlation with lagged {lagged_mode}')
    ax.set_xlabel('lags')
    ax.set_ylabel('correlation')
    fig.savefig(plot_dir + f'lagged_{lagged_mode}_{mode2}.png')

def run_plots(data, mode1, mode2, shifts, plot_dir):
    corr_np = lagged_corr(data=data, mode1=mode1, mode2=mode2, shifts=shifts)
    lagged_mode_plots(corr_np=corr_np,
                      lagged_mode=mode1,
                      mode2=mode2,
                      plot_dir=plot_dir)



run_plots(ortho_mode_df,
  mode1='nao',
  mode2='ao',
  shifts=np.arange(-300, 300),
  plot_dir='/home/data/projects/conus_precip_extremes/plots/mode_coherence/')

pdo_pna_corrs = lagged_corr(data=ortho_mode_df,
  mode1='pdo',
  mode2='pna',
  shifts=np.arange(-300, 300))

np.argmax(np.abs(pdo_pna_corrs))
pdo_pna_corrs[297:305]
pdo_pna_corrs[302]
np.arange(-300, 300)[302]


corr_freq, corr_Px = welch(enso_pdo_corrs, fs=12)

fig, ax = plt.subplots()
ax.plot(corr_freq, corr_Px)
1 / corr_freq[np.argmax(corr_Px)]

fig, ax = plt.subplots()
for i in range(12):
    ax.plot(np.arange(-50, 50), enso_pdo_monthly_corrs[i], alpha=0.3)
plt.close()

enso_pdo_corrs[9]