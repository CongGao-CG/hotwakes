import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import subprocess
from scipy import stats
from pathlib import Path
from diptest import diptest
from collections import Counter
from sst_loader import load_windows
from doy import get_day_of_year_365

data, name = load_windows(Path('t_data'), with_date=False, with_name=True, only_TSHU_status = False)
bdata = data[:, 5:12].mean(axis=1)
data = data[:, 11:20]
name = name.reset_index(drop=True)

# duplicate CP022022 and CP032002
name = name.drop(index=range(21167, 21188)).drop(index=range(21489, 21532))
# only 0 6 12 18
name = name.loc[(name['time'].dt.hour.isin([0, 6, 12, 18]) & name['time'].dt.minute.eq(0) & name['time'].dt.second.eq(0))]
# calculate time difference
name['dt'] = name.groupby('name')['time'].diff().dt.total_seconds().div(3600).astype('float64')
# drop dt = 246
name = name.drop(index=range(45522, 45523))
# drop dt = 24
name = name.drop(index=range(64799, 64800))

name['bgen'] = (name['wind'] >= 35)
name['bgen'] = (name.groupby('name')['bgen'].cumsum() == 0)
name = name[name['bgen'] == False]
name = name.drop('dt', axis=1)
name['dt'] = name.groupby('name')['time'].diff().dt.total_seconds().div(3600).astype('float64')

name['lon'] = (name['lon'] + 360) % 360
name['lon'] = round(name['lon'], 1)
name['lon_grid'] = round((name['lon'] + 0.125) * 4) / 4 - 0.125
name['lon_grid'] = abs(name['lon_grid'])
name['lat_grid'] = round((name['lat'] + 0.125) * 4) / 4 - 0.125
name['doy0'] = get_day_of_year_365(name['time'])
name['doy-4'] = (name['doy0'] - 4 + 365) % 365
name['doy-3'] = (name['doy0'] - 3 + 365) % 365
name['doy-2'] = (name['doy0'] - 2 + 365) % 365
name['doy-1'] = (name['doy0'] - 1 + 365) % 365
name['doy1'] = (name['doy0'] + 1 + 365) % 365
name['doy2'] = (name['doy0'] + 2 + 365) % 365
name['doy3'] = (name['doy0'] + 3 + 365) % 365
name['doy4'] = (name['doy0'] + 4 + 365) % 365

name['LMI'] = name.groupby('name')['wind'].transform('max')
columns = ['name', 'lon', 'lat', 'lon_grid', 'lat_grid', 'time', 'dt', 'wind', 'LMI',
           'doy-4', 'doy-3', 'doy-2', 'doy-1', 'doy0', 'doy1', 'doy2', 'doy3', 'doy4']
name = name[columns]

clm_90_b8211 = xr.open_dataset('/Volumes/Back4SJTU/OISST/clm_90_0.1_359.9_-80.1_80.1_1982_2011.nc')['sst_90th_percentile']
clm_90_b9221 = xr.open_dataset('/Volumes/Back4SJTU/OISST/clm_90_0.1_359.9_-80.1_80.1_1992_2021.nc')['sst_90th_percentile']

doy_columns = columns[-9:]
data_clm_90_b8211 = np.zeros(data.shape)
data_clm_90_b9221 = np.zeros(data.shape)
for i, row in name.iterrows():
    lon = row['lon_grid']
    lat = row['lat_grid']
    for j, doy_col in enumerate(doy_columns):
        doy = row[doy_col]
        data_clm_90_b8211[i, j] = clm_90_b8211.sel(lon=lon, lat=lat, day_of_year=doy + 1).values
        data_clm_90_b9221[i, j] = clm_90_b9221.sel(lon=lon, lat=lat, day_of_year=doy + 1).values

r0_b8211 = np.all(data[name.index, 0:5] >= data_clm_90_b8211[name.index, 0:5], axis=1)
r1_b8211 = np.all(data[name.index, 1:6] >= data_clm_90_b8211[name.index, 1:6], axis=1)
r2_b8211 = np.all(data[name.index, 2:7] >= data_clm_90_b8211[name.index, 2:7], axis=1)
r3_b8211 = np.all(data[name.index, 3:8] >= data_clm_90_b8211[name.index, 3:8], axis=1)
r4_b8211 = np.all(data[name.index, 4:9] >= data_clm_90_b8211[name.index, 4:9], axis=1)
r_b8211  = r0_b8211 | r1_b8211 | r2_b8211 | r3_b8211 | r4_b8211
name['MHW_b8211'] = r_b8211
r0_b9221 = np.all(data[name.index, 0:5] >= data_clm_90_b9221[name.index, 0:5], axis=1)
r1_b9221 = np.all(data[name.index, 1:6] >= data_clm_90_b9221[name.index, 1:6], axis=1)
r2_b9221 = np.all(data[name.index, 2:7] >= data_clm_90_b9221[name.index, 2:7], axis=1)
r3_b9221 = np.all(data[name.index, 3:8] >= data_clm_90_b9221[name.index, 3:8], axis=1)
r4_b9221 = np.all(data[name.index, 4:9] >= data_clm_90_b9221[name.index, 4:9], axis=1)
r_b9221  = r0_b9221 | r1_b9221 | r2_b9221 | r3_b9221 | r4_b9221
name['MHW_b9221'] = r_b9221

name['eLMI'] = (name['wind'] == name['LMI'])
name['bLMI'] = (name.groupby('name')['eLMI'].cumsum() == 0)

name['IC'] = name['LMI'] - name.groupby('name')['wind'].transform('first')
name['IT'] = name.groupby('name')['dt'].transform(lambda x: x[name.loc[x.index, 'bLMI'].shift(1) == True].sum())
name.loc[(name['IT'] == 0), 'IT'] = np.nan
name['IR'] = name['IC'] / name['IT'] * 4
name['bSST'] = bdata[name.index] 
name['cSST'] = data[name.index, 4] 
name['Cooling'] = name['cSST'] - name['bSST']
name['clm_90_b8211'] = data_clm_90_b8211[name.index, 4]
name['clm_90_b9221'] = data_clm_90_b9221[name.index, 4]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7))
beryl_data = name[name['name'] == 'AL022024'][0:25]
beryl_data['MHWi'] = beryl_data['cSST'] - beryl_data['clm_90_b8211']
beryl_data.loc[beryl_data['MHW_b8211'] == False, 'MHWi'] = np.nan
ax1.plot(beryl_data['time'], beryl_data['wind'], 'k.-', markersize=8, linewidth=1)
# ax1.set_xlabel('Time')
ax1.set_ylabel('TC Intensity (knots)')
ax1.set_title('Hurricane Beryl (2024)')
ax1.text(0.02, 0.98, 'a', transform=ax1.transAxes, fontsize=12, fontweight='bold',
         verticalalignment='top')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_locator(mdates.DayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax1_temp = ax1.twinx()
ax1_temp.set_zorder(0)
ax1.set_zorder(1)
ax1.patch.set_alpha(0)
ax1_temp.bar(beryl_data['time'], beryl_data['MHWi'], 
             color='red', alpha=0.6, width=0.1)
ax1_temp.set_ylabel('MHW intensity (°C)', color='red')
ax1_temp.tick_params(axis='y', labelcolor='red', colors='red')
ax1_temp.spines['right'].set_color('red')
milton_data = name[name['name'] == 'AL142024'][0:18]
milton_data['MHWi'] = milton_data['cSST'] - milton_data['clm_90_b8211']
milton_data.loc[milton_data['MHW_b8211'] == False, 'MHWi'] = np.nan
ax2.plot(milton_data['time'], milton_data['wind'], 'k.-', markersize=8, linewidth=1)
# ax2.set_xlabel('Time')
ax2.set_ylabel('TC Intensity (knots)')
ax2.set_title('Hurricane Milton (2024)')
ax2.text(0.02, 0.98, 'b', transform=ax2.transAxes, fontsize=12, fontweight='bold',
         verticalalignment='top')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_locator(mdates.DayLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax2_temp = ax2.twinx()
ax2_temp.set_zorder(0)
ax2.set_zorder(1)
ax2.patch.set_alpha(0)
ax2_temp.bar(milton_data['time'], milton_data['MHWi'], 
             color='red', alpha=0.6, width=0.1)
ax2_temp.set_ylabel('MHW intensity(°C)', color='red')
ax2_temp.tick_params(axis='y', labelcolor='red', colors='red')
ax2_temp.spines['right'].set_color('red')
plt.tight_layout()
output_file = "mhw_plot/Fig1.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])

if input("Continue? (yes/no): ").lower() != 'yes':
    exit()

has_mhw_b8211 = name[(name['bLMI'] == True) & (abs(name['lat']) <= 30)].groupby('name')['MHW_b8211'].sum()
name_wtmhw_b8211 = name[name['name'].isin(has_mhw_b8211[has_mhw_b8211 > 0].index)]
name_nomhw_b8211 = name[~name['name'].isin(has_mhw_b8211[has_mhw_b8211 > 0].index)]

has_mhw_b9221 = name[(name['bLMI'] == True) & (abs(name['lat']) <= 30)].groupby('name')['MHW_b9221'].sum()
name_wtmhw_b9221 = name[name['name'].isin(has_mhw_b9221[has_mhw_b9221 > 0].index)]
name_nomhw_b9221 = name[~name['name'].isin(has_mhw_b9221[has_mhw_b9221 > 0].index)]

lmi_wtmhw_b8211 = name_wtmhw_b8211.groupby('name')['LMI'].max()
lmi_nomhw_b8211 = name_nomhw_b8211.groupby('name')['LMI'].max()

lmi_wtmhw_b9221 = name_wtmhw_b9221.groupby('name')['LMI'].max()
lmi_nomhw_b9221 = name_nomhw_b9221.groupby('name')['LMI'].max()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7))
x = np.linspace(0, 200, 200)
density_wtmhw_b8211 = stats.gaussian_kde(lmi_wtmhw_b8211.dropna())
ax1.plot(x, density_wtmhw_b8211(x), 'r-', linewidth=2, 
         label='With-MHW', alpha=0.8)
density_nomhw_b8211 = stats.gaussian_kde(lmi_nomhw_b8211.dropna())
ax1.plot(x, density_nomhw_b8211(x), 'b-', linewidth=2, 
         label='No-MHW', alpha=0.8)
ax1.set_xlabel('Lifetime maximum intensity (knots)')
ax1.set_ylabel('Probability density')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_title('GL: MHW baseline 1982–2011')
ax1.text(0.02, 0.98, 'a', transform=ax1.transAxes, fontsize=12, fontweight='bold',
         verticalalignment='top')
density_wtmhw_b9221 = stats.gaussian_kde(lmi_wtmhw_b9221.dropna())
ax2.plot(x, density_wtmhw_b9221(x), 'r-', linewidth=2,
         label='With-MHW', alpha=0.8)
density_nomhw_b9221 = stats.gaussian_kde(lmi_nomhw_b9221.dropna())
ax2.plot(x, density_nomhw_b9221(x), 'b-', linewidth=2,
         label='No-MHW', alpha=0.8)
ax2.set_xlabel('Lifetime maximum intensity (knots)')
ax2.set_ylabel('Probability density')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_title('GL: MHW baseline: 1992–2021')
ax2.text(0.02, 0.98, 'b', transform=ax2.transAxes, fontsize=12, fontweight='bold',
         verticalalignment='top')
plt.tight_layout()
output_file = "mhw_plot/Fig2.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])
stats.ks_2samp(lmi_nomhw_b8211.dropna(), lmi_wtmhw_b8211.dropna())
stats.ks_2samp(lmi_nomhw_b9221.dropna(), lmi_wtmhw_b9221.dropna())

fig, axes = plt.subplots(2, 3, figsize=(10, 5))
axes = axes.flatten()
basins = [
    ('AL', 'Atlantic', lambda x: x.str.startswith('AL')),
    ('EP', 'Eastern Pacific', lambda x: x.str.startswith('EP') | x.str.startswith('CP')),
    ('SH', 'Southern Hemisphere', lambda x: x.str.startswith('SH')),
    ('IO', 'Indian Ocean', lambda x: x.str.startswith('IO')),
    ('WP', 'Western Pacific', lambda x: x.str.startswith('WP'))
]
subplot_labels = ['a', 'b', 'c', 'd', 'e']
for idx, (basin_code, basin_name, filter_func) in enumerate(basins):
    ax = axes[idx]
    basin_wtmhw = name_wtmhw_b8211[filter_func(name_wtmhw_b8211['name'])]
    basin_nomhw = name_nomhw_b8211[filter_func(name_nomhw_b8211['name'])]
    lmi_wtmhw = basin_wtmhw.groupby('name')['LMI'].max()
    lmi_nomhw = basin_nomhw.groupby('name')['LMI'].max()
    x = np.linspace(0, 200, 200)
    if len(lmi_wtmhw.dropna()) > 0:
        density_wtmhw = stats.gaussian_kde(lmi_wtmhw.dropna())
        ax.plot(x, density_wtmhw(x), 'r-', linewidth=2,
                label='With-MHW', alpha=0.8)
    if len(lmi_nomhw.dropna()) > 0:
        density_nomhw = stats.gaussian_kde(lmi_nomhw.dropna())
        ax.plot(x, density_nomhw(x), 'b-', linewidth=2,
                label='No-MHW', alpha=0.8)
    ax.set_xlabel('Lifetime maximum intensity (knots)')
    ax.set_ylabel('Probability density')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{basin_code}: MHW baseline 1982–2011')
    ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes, 
            fontsize=12, fontweight='bold', verticalalignment='top')
    stats.ks_2samp(lmi_nomhw.dropna(), lmi_wtmhw.dropna())


fig.delaxes(axes[5])
plt.tight_layout()
output_file = "mhw_plot/FigS1.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])

fig, axes = plt.subplots(2, 3, figsize=(10, 5))
axes = axes.flatten()
for idx, (basin_code, basin_name, filter_func) in enumerate(basins):
    ax = axes[idx]
    basin_wtmhw = name_wtmhw_b9221[filter_func(name_wtmhw_b9221['name'])]
    basin_nomhw = name_nomhw_b9221[filter_func(name_nomhw_b9221['name'])]
    lmi_wtmhw = basin_wtmhw.groupby('name')['LMI'].max()
    lmi_nomhw = basin_nomhw.groupby('name')['LMI'].max()
    x = np.linspace(0, 200, 200)
    if len(lmi_wtmhw.dropna()) > 0:
        density_wtmhw = stats.gaussian_kde(lmi_wtmhw.dropna())
        ax.plot(x, density_wtmhw(x), 'r-', linewidth=2,
                label='With-MHW', alpha=0.8)
    if len(lmi_nomhw.dropna()) > 0:
        density_nomhw = stats.gaussian_kde(lmi_nomhw.dropna())
        ax.plot(x, density_nomhw(x), 'b-', linewidth=2,
                label='No-MHW', alpha=0.8)
    ax.set_xlabel('Lifetime maximum intensity (knots)')
    ax.set_ylabel('Probability density')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{basin_code}: MHW baseline 1992–2021')
    ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top')
    stats.ks_2samp(lmi_nomhw.dropna(), lmi_wtmhw.dropna())


fig.delaxes(axes[5])
plt.tight_layout()
output_file = "mhw_plot/FigS2.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])

fig, axes = plt.subplots(2, 2, figsize=(6, 9))
axes = axes.flatten()
x_pos = [0, 1]
colors   = ['blue', 'red']
nomhw_se = lmi_nomhw_b8211.dropna().values.std() / np.sqrt(len(lmi_nomhw_b8211.dropna().values))
wtmhw_se = lmi_wtmhw_b8211.dropna().values.std() / np.sqrt(len(lmi_wtmhw_b8211.dropna().values))
means   = [lmi_nomhw_b8211.dropna().values.mean(), lmi_wtmhw_b8211.dropna().values.mean()]
errors   = [nomhw_se, wtmhw_se]
bars = axes[0].bar(x_pos, means, yerr=errors, capsize=10, color=colors, 
                   alpha=0.7, edgecolor='black', linewidth=0.5, width=0.4)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(['No-MHW', 'With-MHW'])
axes[0].set_ylabel('Lifetime maximum intensity (knots)')
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_title('GL: MHW baseline 1982–2011')
axes[0].text(0.02, 0.98, 'a', transform=axes[0].transAxes, fontsize=12, fontweight='bold',
         verticalalignment='top')
for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + err,
                 f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

nomhw_se = lmi_nomhw_b9221.dropna().values.std() / np.sqrt(len(lmi_nomhw_b9221.dropna().values))
wtmhw_se = lmi_wtmhw_b9221.dropna().values.std() / np.sqrt(len(lmi_wtmhw_b9221.dropna().values))
means   = [lmi_nomhw_b9221.dropna().values.mean(), lmi_wtmhw_b9221.dropna().values.mean()]
errors   = [nomhw_se, wtmhw_se]
bars = axes[1].bar(x_pos, means, yerr=errors, capsize=10, color=colors,                
                   alpha=0.7, edgecolor='black', linewidth=0.5, width=0.4)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(['No-MHW', 'With-MHW'])
axes[1].set_ylabel('Lifetime maximum intensity (knots)')
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_title('GL: MHW baseline: 1992–2021')
axes[1].text(0.02, 0.98, 'b', transform=axes[1].transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top')
for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + err,
                 f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ic_nomhw_b8211 = name_nomhw_b8211.groupby('name')['IC'].max()
ic_wtmhw_b8211 = name_wtmhw_b8211.groupby('name')['IC'].max()
nomhw_se = ic_nomhw_b8211.dropna().values.std() / np.sqrt(len(ic_nomhw_b8211.dropna().values))
wtmhw_se = ic_wtmhw_b8211.dropna().values.std() / np.sqrt(len(ic_wtmhw_b8211.dropna().values))
means   = [ic_nomhw_b8211.dropna().values.mean(), ic_wtmhw_b8211.dropna().values.mean()]
errors   = [nomhw_se, wtmhw_se]
bars = axes[2].bar(x_pos, means, yerr=errors, capsize=10, color=colors,
                   alpha=0.7, edgecolor='black', linewidth=0.5, width=0.4)
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(['No-MHW', 'With-MHW'])
axes[2].set_ylabel('Intensity change (knots)')
axes[2].grid(axis='y', alpha=0.3)
axes[2].set_title('GL: MHW baseline 1982–2011')
axes[2].text(0.02, 0.98, 'c', transform=axes[2].transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top')
for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
    height = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2., height + err,
                 f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ic_nomhw_b9221 = name_nomhw_b9221.groupby('name')['IC'].max()
ic_wtmhw_b9221 = name_wtmhw_b9221.groupby('name')['IC'].max()
nomhw_se = ic_nomhw_b9221.dropna().values.std() / np.sqrt(len(ic_nomhw_b9221.dropna().values))
wtmhw_se = ic_wtmhw_b9221.dropna().values.std() / np.sqrt(len(ic_wtmhw_b9221.dropna().values))
means   = [ic_nomhw_b9221.dropna().values.mean(), ic_wtmhw_b9221.dropna().values.mean()]
errors   = [nomhw_se, wtmhw_se]
bars = axes[3].bar(x_pos, means, yerr=errors, capsize=10, color=colors,
                   alpha=0.7, edgecolor='black', linewidth=0.5, width=0.4)
axes[3].set_xticks(x_pos)
axes[3].set_xticklabels(['No-MHW', 'With-MHW'])
axes[3].set_ylabel('Intensity change (knots)')
axes[3].grid(axis='y', alpha=0.3)
axes[3].set_title('GL: MHW baseline 1992–2021')
axes[3].text(0.02, 0.98, 'd', transform=axes[3].transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top')
for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
    height = bar.get_height()
    axes[3].text(bar.get_x() + bar.get_width()/2., height + err,
                 f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
output_file = "mhw_plot/Fig3.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])

fig, axes = plt.subplots(2, 3, figsize=(10, 9))
axes = axes.flatten()
for idx, (basin_code, basin_name, filter_func) in enumerate(basins):
    ax = axes[idx]
    basin_wtmhw = name_wtmhw_b8211[filter_func(name_wtmhw_b8211['name'])]
    basin_nomhw = name_nomhw_b8211[filter_func(name_nomhw_b8211['name'])]
    lmi_wtmhw = basin_wtmhw.groupby('name')['LMI'].max()
    lmi_nomhw = basin_nomhw.groupby('name')['LMI'].max()
    nomhw_se = lmi_nomhw.dropna().values.std() / np.sqrt(len(lmi_nomhw.dropna().values))
    wtmhw_se = lmi_wtmhw.dropna().values.std() / np.sqrt(len(lmi_wtmhw.dropna().values))
    means    = [lmi_nomhw.dropna().values.mean(), lmi_wtmhw.dropna().values.mean()]
    errors   = [nomhw_se, wtmhw_se]
    bars = ax.bar(x_pos, means, yerr=errors, capsize=10, color=colors,
                  alpha=0.7, edgecolor='black', linewidth=0.5, width=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['No-MHW', 'With-MHW'])
    ax.set_ylabel('Lifetime maximum intensity (knots)')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'{basin_code}: MHW baseline 1982–2011')
    ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top')
    for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')



fig.delaxes(axes[5])
plt.tight_layout()
output_file = "mhw_plot/FigS3.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])

fig, axes = plt.subplots(2, 3, figsize=(10, 9))
axes = axes.flatten()
for idx, (basin_code, basin_name, filter_func) in enumerate(basins):
    ax = axes[idx]
    basin_wtmhw = name_wtmhw_b9221[filter_func(name_wtmhw_b9221['name'])]
    basin_nomhw = name_nomhw_b9221[filter_func(name_nomhw_b9221['name'])]
    lmi_wtmhw = basin_wtmhw.groupby('name')['LMI'].max()
    lmi_nomhw = basin_nomhw.groupby('name')['LMI'].max()
    nomhw_se = lmi_nomhw.dropna().values.std() / np.sqrt(len(lmi_nomhw.dropna().values))
    wtmhw_se = lmi_wtmhw.dropna().values.std() / np.sqrt(len(lmi_wtmhw.dropna().values))
    means    = [lmi_nomhw.dropna().values.mean(), lmi_wtmhw.dropna().values.mean()]
    errors   = [nomhw_se, wtmhw_se]
    bars = ax.bar(x_pos, means, yerr=errors, capsize=10, color=colors,
                  alpha=0.7, edgecolor='black', linewidth=0.5, width=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['No-MHW', 'With-MHW'])
    ax.set_ylabel('Lifetime maximum intensity (knots)')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'{basin_code}: MHW baseline 1992–2021')
    ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top')
    for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')



fig.delaxes(axes[5])
plt.tight_layout()
output_file = "mhw_plot/FigS4.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])

fig, axes = plt.subplots(2, 3, figsize=(10, 9))
axes = axes.flatten()
for idx, (basin_code, basin_name, filter_func) in enumerate(basins):
    ax = axes[idx]
    basin_wtmhw = name_wtmhw_b8211[filter_func(name_wtmhw_b8211['name'])]
    basin_nomhw = name_nomhw_b8211[filter_func(name_nomhw_b8211['name'])]
    ic_wtmhw = basin_wtmhw.groupby('name')['IC'].max()
    ic_nomhw = basin_nomhw.groupby('name')['IC'].max()
    nomhw_se = ic_nomhw.dropna().values.std() / np.sqrt(len(ic_nomhw.dropna().values))
    wtmhw_se = ic_wtmhw.dropna().values.std() / np.sqrt(len(ic_wtmhw.dropna().values))
    means    = [ic_nomhw.dropna().values.mean(), ic_wtmhw.dropna().values.mean()]
    errors   = [nomhw_se, wtmhw_se]
    bars = ax.bar(x_pos, means, yerr=errors, capsize=10, color=colors,
                  alpha=0.7, edgecolor='black', linewidth=0.5, width=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['No-MHW', 'With-MHW'])
    ax.set_ylabel('Intensity change (knots)')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'{basin_code}: MHW baseline 1982–2011')
    ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top')
    for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')



fig.delaxes(axes[5])
plt.tight_layout()
output_file = "mhw_plot/FigS5.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])

fig, axes = plt.subplots(2, 3, figsize=(10, 9))
axes = axes.flatten()
for idx, (basin_code, basin_name, filter_func) in enumerate(basins):
    ax = axes[idx]
    basin_wtmhw = name_wtmhw_b9221[filter_func(name_wtmhw_b9221['name'])]
    basin_nomhw = name_nomhw_b9221[filter_func(name_nomhw_b9221['name'])]
    ic_wtmhw = basin_wtmhw.groupby('name')['IC'].max()
    ic_nomhw = basin_nomhw.groupby('name')['IC'].max()
    nomhw_se = ic_nomhw.dropna().values.std() / np.sqrt(len(ic_nomhw.dropna().values))
    wtmhw_se = ic_wtmhw.dropna().values.std() / np.sqrt(len(ic_wtmhw.dropna().values))
    means    = [ic_nomhw.dropna().values.mean(), ic_wtmhw.dropna().values.mean()]
    errors   = [nomhw_se, wtmhw_se]
    bars = ax.bar(x_pos, means, yerr=errors, capsize=10, color=colors,
                  alpha=0.7, edgecolor='black', linewidth=0.5, width=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['No-MHW', 'With-MHW'])
    ax.set_ylabel('Intensity change (knots)')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'{basin_code}: MHW baseline 1992–2021')
    ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top')
    for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')



fig.delaxes(axes[5])
plt.tight_layout()
output_file = "mhw_plot/FigS6.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])

"""
ir_nomhw = name_nomhw.groupby('name')['IR'].max()
ir_wtmhw = name_wtmhw.groupby('name')['IR'].max()
fig, ax = plt.subplots(figsize=(5, 6))
nomhw_se = ir_nomhw.dropna().values.std() / np.sqrt(len(ir_nomhw.dropna().values))
wtmhw_se = ir_wtmhw.dropna().values.std() / np.sqrt(len(ir_wtmhw.dropna().values))
x_pos = [0, 1]
means = [ir_nomhw.dropna().values.mean(), ir_wtmhw.dropna().values.mean()]
errors = [nomhw_se, wtmhw_se]
colors = ['blue', 'red']
bars = ax.bar(x_pos, means, yerr=errors, capsize=10, color=colors,
               alpha=0.7, edgecolor='black', linewidth=2)
ax.set_xticks(x_pos)
ax.set_xticklabels(['No MHW', 'With MHW'])
ax.set_ylabel('Intensification rate (kt/day)', fontsize=12)
ax.grid(axis='y', alpha=0.3)
for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + err,
            f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
"""

def resample_group(group):
    group = group.set_index('time')
    resampled = group.resample('6h').asfreq()
    resampled['name'] = resampled['name'].ffill()
    resampled['wind'] = resampled['wind'].interpolate(method='linear')
    return resampled.reset_index()

name_ri = name[['name', 'time', 'wind']].groupby('name', group_keys=False).apply(resample_group).reset_index(drop=True)
name_ri['dt'] = name_ri.groupby('name')['time'].diff().dt.total_seconds().div(3600).astype('float64')
name_ri['IR24'] = name_ri.groupby('name')['wind'].transform(lambda x: x.shift(-2) - x.shift(2))
name_ri['RI'] = (name_ri['IR24'] >= 30)
name_ri['RIf'] = name_ri.groupby('name')['RI'].transform('sum')

name_ri_wtmhw_b8211 = name_ri[name_ri['name'].isin(has_mhw_b8211[has_mhw_b8211 > 0].index)]
name_ri_nomhw_b8211 = name_ri[~name_ri['name'].isin(has_mhw_b8211[has_mhw_b8211 > 0].index)]

name_ri_wtmhw_b9221 = name_ri[name_ri['name'].isin(has_mhw_b9221[has_mhw_b9221 > 0].index)]
name_ri_nomhw_b9221 = name_ri[~name_ri['name'].isin(has_mhw_b9221[has_mhw_b9221 > 0].index)]

ri_wtmhw_b8211 = [name_ri_wtmhw_b8211[name_ri_wtmhw_b8211['RIf'] > 0]['name'].unique().shape[0], name_ri_wtmhw_b8211[name_ri_wtmhw_b8211['RIf'] == 0]['name'].unique().shape[0]]
ri_nomhw_b8211 = [name_ri_nomhw_b8211[name_ri_nomhw_b8211['RIf'] > 0]['name'].unique().shape[0], name_ri_nomhw_b8211[name_ri_nomhw_b8211['RIf'] == 0]['name'].unique().shape[0]]
ratio_wtmhw_b8211 = ri_wtmhw_b8211[0] / sum(ri_wtmhw_b8211)
ratio_nomhw_b8211 = ri_nomhw_b8211[0] / sum(ri_nomhw_b8211)

ri_wtmhw_b9221 = [name_ri_wtmhw_b9221[name_ri_wtmhw_b9221['RIf'] > 0]['name'].unique().shape[0], name_ri_wtmhw_b9221[name_ri_wtmhw_b9221['RIf'] == 0]['name'].unique().shape[0]]
ri_nomhw_b9221 = [name_ri_nomhw_b9221[name_ri_nomhw_b9221['RIf'] > 0]['name'].unique().shape[0], name_ri_nomhw_b9221[name_ri_nomhw_b9221['RIf'] == 0]['name'].unique().shape[0]]
ratio_wtmhw_b9221 = ri_wtmhw_b9221[0] / sum(ri_wtmhw_b9221)
ratio_nomhw_b9221 = ri_nomhw_b9221[0] / sum(ri_nomhw_b9221)

fig, axes = plt.subplots(2, 2, figsize=(7, 9))
axes = axes.flatten()

wtmhw_se = np.sqrt(ratio_wtmhw_b8211 * (1 - ratio_wtmhw_b8211) / sum(ri_wtmhw_b8211))
nomhw_se = np.sqrt(ratio_nomhw_b8211 * (1 - ratio_nomhw_b8211) / sum(ri_nomhw_b8211))
means =  [ratio_nomhw_b8211, ratio_wtmhw_b8211]
errors = [nomhw_se, wtmhw_se]
bars = axes[0].bar(x_pos, means, yerr=errors, capsize=10, color=colors,
                   alpha=0.7, edgecolor='black', linewidth=0.5, width=0.4)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(['No-MHW', 'With-MHW'])
axes[0].set_ylabel('Ratio of rapid-intensification TCs')
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_title('GL: MHW baseline 1982–2011')
axes[0].text(0.02, 0.98, 'a', transform=axes[0].transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top')
for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + err,
                 f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

wtmhw_se = np.sqrt(ratio_wtmhw_b9221 * (1 - ratio_wtmhw_b9221) / sum(ri_wtmhw_b9221))
nomhw_se = np.sqrt(ratio_nomhw_b9221 * (1 - ratio_nomhw_b9221) / sum(ri_nomhw_b9221))
means =  [ratio_nomhw_b9221, ratio_wtmhw_b9221]
errors = [nomhw_se, wtmhw_se]
bars = axes[1].bar(x_pos, means, yerr=errors, capsize=10, color=colors,
                   alpha=0.7, edgecolor='black', linewidth=0.5, width=0.4)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(['No-MHW', 'With-MHW']) 
axes[1].set_ylabel('Ratio of rapid-intensification TCs')
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_title('GL: MHW baseline 1992–2021')
axes[1].text(0.02, 0.98, 'b', transform=axes[1].transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top')
for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + err,
                 f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

cl_nomhw_b8211 = name_nomhw_b8211[name_nomhw_b8211['bLMI']].groupby('name')['Cooling'].mean()
cl_wtmhw_b8211 = name_wtmhw_b8211[name_wtmhw_b8211['bLMI']].groupby('name')['Cooling'].mean()
nomhw_se = cl_nomhw_b8211.dropna().values.std() / np.sqrt(len(cl_nomhw_b8211.dropna().values))
wtmhw_se = cl_wtmhw_b8211.dropna().values.std() / np.sqrt(len(cl_wtmhw_b8211.dropna().values))
means = [cl_nomhw_b8211.dropna().values.mean(), cl_wtmhw_b8211.dropna().values.mean()]
errors = [nomhw_se, wtmhw_se]
bars = axes[2].bar(x_pos, means, yerr=errors, capsize=10, color=colors,
                   alpha=0.7, edgecolor='black', linewidth=0.5, width=0.4)
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(['No-MHW', 'With-MHW'])
axes[2].set_ylabel('Sea surface temperature cooling (°C)')
axes[2].grid(axis='y', alpha=0.3)
axes[2].set_title('GL: MHW baseline 1982–2011')
axes[2].text(0.02, 0.98, 'c', transform=axes[2].transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='square', facecolor='white', alpha=1, linewidth=0, pad=0.1))
for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
    height = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2., height - err,
                 f'{mean:.2f}', ha='center', va='top', fontsize=12, fontweight='bold')

cl_nomhw_b9221 = name_nomhw_b9221[name_nomhw_b9221['bLMI']].groupby('name')['Cooling'].mean()
cl_wtmhw_b9221 = name_wtmhw_b9221[name_wtmhw_b9221['bLMI']].groupby('name')['Cooling'].mean()
nomhw_se = cl_nomhw_b9221.dropna().values.std() / np.sqrt(len(cl_nomhw_b9221.dropna().values))
wtmhw_se = cl_wtmhw_b9221.dropna().values.std() / np.sqrt(len(cl_wtmhw_b9221.dropna().values))
means = [cl_nomhw_b9221.dropna().values.mean(), cl_wtmhw_b9221.dropna().values.mean()]
errors = [nomhw_se, wtmhw_se]
bars = axes[3].bar(x_pos, means, yerr=errors, capsize=10, color=colors,
                   alpha=0.7, edgecolor='black', linewidth=0.5, width=0.4)
axes[3].set_xticks(x_pos)
axes[3].set_xticklabels(['No-MHW', 'With-MHW'])
axes[3].set_ylabel('Sea surface temperature cooling (°C)')
axes[3].grid(axis='y', alpha=0.3)
axes[3].set_title('GL: MHW baseline 1992–2021')
axes[3].text(0.02, 0.98, 'd', transform=axes[3].transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='square', facecolor='white', alpha=1, linewidth=0, pad=0.1))
for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
    height = bar.get_height()
    axes[3].text(bar.get_x() + bar.get_width()/2., height - err,
                 f'{mean:.2f}', ha='center', va='top', fontsize=12, fontweight='bold')

plt.tight_layout()
output_file = "mhw_plot/Fig4.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])

fig, axes = plt.subplots(2, 3, figsize=(10, 9))
axes = axes.flatten()
for idx, (basin_code, basin_name, filter_func) in enumerate(basins):
    ax = axes[idx]
    basin_wtmhw = name_ri_wtmhw_b8211[filter_func(name_ri_wtmhw_b8211['name'])]
    basin_nomhw = name_ri_nomhw_b8211[filter_func(name_ri_nomhw_b8211['name'])]
    ri_wtmhw = [basin_wtmhw[basin_wtmhw['RIf'] > 0]['name'].unique().shape[0], basin_wtmhw[basin_wtmhw['RIf'] == 0]['name'].unique().shape[0]]
    ri_nomhw = [basin_nomhw[basin_nomhw['RIf'] > 0]['name'].unique().shape[0], basin_nomhw[basin_nomhw['RIf'] == 0]['name'].unique().shape[0]]
    ratio_wtmhw = ri_wtmhw[0] / sum(ri_wtmhw)
    ratio_nomhw = ri_nomhw[0] / sum(ri_nomhw)
    wtmhw_se = np.sqrt(ratio_wtmhw * (1 - ratio_wtmhw) / sum(ri_wtmhw))
    nomhw_se = np.sqrt(ratio_nomhw * (1 - ratio_nomhw) / sum(ri_nomhw))
    means = [ratio_nomhw, ratio_wtmhw]
    errors = [nomhw_se, wtmhw_se]
    bars = ax.bar(x_pos, means, yerr=errors, capsize=10, color=colors,
                  alpha=0.7, edgecolor='black', linewidth=0.5, width=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['No-MHW', 'With-MHW'])
    ax.set_ylabel('Ratio of rapid-intensification TCs')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'{basin_code}: MHW baseline 1982–2011')
    ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top')
    for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')



fig.delaxes(axes[5])
plt.tight_layout()
output_file = "mhw_plot/FigS7.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])

fig, axes = plt.subplots(2, 3, figsize=(10, 9))
axes = axes.flatten()
for idx, (basin_code, basin_name, filter_func) in enumerate(basins):
    ax = axes[idx]
    basin_wtmhw = name_ri_wtmhw_b9221[filter_func(name_ri_wtmhw_b9221['name'])]
    basin_nomhw = name_ri_nomhw_b9221[filter_func(name_ri_nomhw_b9221['name'])]
    ri_wtmhw = [basin_wtmhw[basin_wtmhw['RIf'] > 0]['name'].unique().shape[0], basin_wtmhw[basin_wtmhw['RIf'] == 0]['name'].unique().shape[0]]
    ri_nomhw = [basin_nomhw[basin_nomhw['RIf'] > 0]['name'].unique().shape[0], basin_nomhw[basin_nomhw['RIf'] == 0]['name'].unique().shape[0]]
    ratio_wtmhw = ri_wtmhw[0] / sum(ri_wtmhw)
    ratio_nomhw = ri_nomhw[0] / sum(ri_nomhw)
    wtmhw_se = np.sqrt(ratio_wtmhw * (1 - ratio_wtmhw) / sum(ri_wtmhw))
    nomhw_se = np.sqrt(ratio_nomhw * (1 - ratio_nomhw) / sum(ri_nomhw))
    means = [ratio_nomhw, ratio_wtmhw]
    errors = [nomhw_se, wtmhw_se]
    bars = ax.bar(x_pos, means, yerr=errors, capsize=10, color=colors,
                  alpha=0.7, edgecolor='black', linewidth=0.5, width=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['No-MHW', 'With-MHW'])
    ax.set_ylabel('Ratio of rapid-intensification TCs')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'{basin_code}: MHW baseline 1992–2021')
    ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top')
    for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')



fig.delaxes(axes[5])
plt.tight_layout()
output_file = "mhw_plot/FigS8.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])

fig, axes = plt.subplots(2, 3, figsize=(10, 9))
axes = axes.flatten()
for idx, (basin_code, basin_name, filter_func) in enumerate(basins):
    ax = axes[idx]
    basin_wtmhw = name_wtmhw_b8211[filter_func(name_wtmhw_b8211['name'])]
    basin_nomhw = name_nomhw_b8211[filter_func(name_nomhw_b8211['name'])]
    cl_wtmhw = basin_wtmhw[basin_wtmhw['bLMI']].groupby('name')['Cooling'].mean()
    cl_nomhw = basin_nomhw[basin_nomhw['bLMI']].groupby('name')['Cooling'].mean()
    nomhw_se = cl_nomhw.dropna().values.std() / np.sqrt(len(cl_nomhw.dropna().values))
    wtmhw_se = cl_wtmhw.dropna().values.std() / np.sqrt(len(cl_wtmhw.dropna().values))
    means    = [cl_nomhw.dropna().values.mean(), cl_wtmhw.dropna().values.mean()]
    errors   = [nomhw_se, wtmhw_se]
    bars = ax.bar(x_pos, means, yerr=errors, capsize=10, color=colors,
                  alpha=0.7, edgecolor='black', linewidth=0.5, width=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['No-MHW', 'With-MHW'])
    ax.set_ylabel('Sea surface temperature cooling (°C)')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'{basin_code}: MHW baseline 1982–2011')
    ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top', bbox=dict(boxstyle='square', facecolor='white', alpha=1, linewidth=0, pad=0.1))
    for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height - err,
                f'{mean:.2f}', ha='center', va='top', fontsize=12, fontweight='bold')


    
fig.delaxes(axes[5])
plt.tight_layout()
output_file = "mhw_plot/FigS9.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])

fig, axes = plt.subplots(2, 3, figsize=(10, 9))
axes = axes.flatten()
for idx, (basin_code, basin_name, filter_func) in enumerate(basins):
    ax = axes[idx]
    basin_wtmhw = name_wtmhw_b9221[filter_func(name_wtmhw_b9221['name'])]
    basin_nomhw = name_nomhw_b9221[filter_func(name_nomhw_b9221['name'])]
    cl_wtmhw = basin_wtmhw[basin_wtmhw['bLMI']].groupby('name')['Cooling'].mean()
    cl_nomhw = basin_nomhw[basin_nomhw['bLMI']].groupby('name')['Cooling'].mean()
    nomhw_se = cl_nomhw.dropna().values.std() / np.sqrt(len(cl_nomhw.dropna().values))
    wtmhw_se = cl_wtmhw.dropna().values.std() / np.sqrt(len(cl_wtmhw.dropna().values))
    means    = [cl_nomhw.dropna().values.mean(), cl_wtmhw.dropna().values.mean()]
    errors   = [nomhw_se, wtmhw_se]
    bars = ax.bar(x_pos, means, yerr=errors, capsize=10, color=colors,
                  alpha=0.7, edgecolor='black', linewidth=0.5, width=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['No-MHW', 'With-MHW'])
    ax.set_ylabel('Sea surface temperature cooling (°C)')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'{basin_code}: MHW baseline 1992–2021')
    ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top', bbox=dict(boxstyle='square', facecolor='white', alpha=1, linewidth=0, pad=0.1))
    for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height - err,
                f'{mean:.2f}', ha='center', va='top', fontsize=12, fontweight='bold')



fig.delaxes(axes[5])
plt.tight_layout()
output_file = "mhw_plot/FigS10.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])

fig, axes = plt.subplots(1, 2, figsize=(7, 4.5))
axes = axes.flatten()

x = []
y = []
for i in range(1, 13):
    mean_lmi = name[name['name'].isin(has_mhw_b8211[has_mhw_b8211 == i].index)].groupby('name')['LMI'].max().mean()
    x.append(i)
    y.append(mean_lmi)

x = np.array(x)
y = np.array(y)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(stats.linregress(x, y))
regression_line = slope * x + intercept
axes[0].scatter(x, y, color='black', s=50, zorder=5)
axes[0].plot(x, regression_line, color='black', linestyle='-', linewidth=1.5)
axes[0].set_xlabel('Number of MHW encounters')
axes[0].set_ylabel('Lifetime maximum intensity (knots)')
axes[0].grid(True, alpha=0.3)
axes[0].set_title('GL: MHW baseline 1982–2011')
axes[0].text(0.02, 0.98, 'a', transform=axes[0].transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top')

x = []
y = []
for i in range(1, 13):
    mean_lmi = name[name['name'].isin(has_mhw_b9221[has_mhw_b9221 == i].index)].groupby('name')['LMI'].max().mean()
    x.append(i)
    y.append(mean_lmi)

x = np.array(x)
y = np.array(y)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(stats.linregress(x, y))
regression_line = slope * x + intercept
axes[1].scatter(x, y, color='black', s=50, zorder=5)
axes[1].plot(x, regression_line, color='black', linestyle='-', linewidth=1.5)
axes[1].set_xlabel('Number of MHW encounters')
axes[1].set_ylabel('Lifetime maximum intensity (knots)')
axes[1].grid(True, alpha=0.3)
axes[1].set_title('GL: MHW baseline 1992–2021')
axes[1].text(0.02, 0.98, 'b', transform=axes[1].transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top')

"""
x = []
y = []
for i in range(1, 13):
    mean_ri = name_ri[name_ri['name'].isin(has_mhw_b8211[has_mhw_b8211 == i].index)].groupby('name')['RI'].any().sum() / name_ri[name_ri['name'].isin(has_mhw_b8211[has_mhw_b8211 == i].index)]['name'].unique().shape[0]
    x.append(i)
    y.append(mean_ri)

x = np.array(x)
y = np.array(y)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(stats.linregress(x, y))
regression_line = slope * x + intercept
axes[2].scatter(x, y, color='black', s=50, zorder=5)
axes[2].plot(x, regression_line, color='black', linestyle='-', linewidth=1.5)
axes[2].set_xlabel('Number of MHW encounters')
axes[2].set_ylabel('Ratio of rapid-intensification TCs')
axes[2].grid(True, alpha=0.3)
axes[2].set_title('GL: MHW baseline 1982–2011')
axes[2].text(0.02, 0.98, 'c', transform=axes[2].transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top')

x = []
y = []
for i in range(1, 13):
    mean_ri = name_ri[name_ri['name'].isin(has_mhw_b9221[has_mhw_b9221 == i].index)].groupby('name')['RI'].any().sum() / name_ri[name_ri['name'].isin(has_mhw_b9221[has_mhw_b9221 == i].index)]['name'].unique().shape[0]
    x.append(i)
    y.append(mean_ri)

x = np.array(x)
y = np.array(y)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(stats.linregress(x, y))
regression_line = slope * x + intercept
axes[3].scatter(x, y, color='black', s=50, zorder=5)
axes[3].plot(x, regression_line, color='black', linestyle='-', linewidth=1.5)
axes[3].set_xlabel('Number of MHW encounters')
axes[3].set_ylabel('Ratio of rapid-intensification TCs')
axes[3].grid(True, alpha=0.3)
axes[3].set_title('GL: MHW baseline 1992–2021')
axes[3].text(0.02, 0.98, 'd', transform=axes[3].transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top')
"""
plt.tight_layout()
output_file = "mhw_plot/Fig5.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])

fig, axes = plt.subplots(2, 3, figsize=(10, 9))
axes = axes.flatten()

n_points = np.array([13, 13, 10, 7, 13])
for idx, (basin_code, basin_name, filter_func) in enumerate(basins):
    ax = axes[idx]
    basin_name = name[filter_func(name['name'])]
    x = []
    y = []       
    for i in range(1, n_points[idx]):
        mean_lmi = basin_name[basin_name['name'].isin(has_mhw_b8211[has_mhw_b8211 == i].index)].groupby('name')['LMI'].max().mean()
        x.append(i)
        y.append(mean_lmi)
    x = np.array(x)
    y = np.array(y)
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(stats.linregress(x, y))
    regression_line = slope * x + intercept
    ax.scatter(x, y, color='black', s=50, zorder=5)
    ax.plot(x, regression_line, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel('Number of MHW encounters')
    ax.set_ylabel('Lifetime maximum intensity (knots)')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{basin_code}: MHW baseline 1982–2011')
    ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top')


fig.delaxes(axes[5])
plt.tight_layout()
output_file = "mhw_plot/FigS11.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])

fig, axes = plt.subplots(2, 3, figsize=(10, 9))
axes = axes.flatten()

n_points = np.array([13, 9, 7, 5, 10])
for idx, (basin_code, basin_name, filter_func) in enumerate(basins):
    ax = axes[idx]
    basin_name = name[filter_func(name['name'])]
    x = []
    y = []
    for i in range(1, n_points[idx]):
        mean_lmi = basin_name[basin_name['name'].isin(has_mhw_b9221[has_mhw_b9221 == i].index)].groupby('name')['LMI'].max().mean()
        x.append(i)
        y.append(mean_lmi)
    x = np.array(x)
    y = np.array(y)
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(stats.linregress(x, y))
    regression_line = slope * x + intercept
    ax.scatter(x, y, color='black', s=50, zorder=5)
    ax.plot(x, regression_line, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel('Number of MHW encounters')
    ax.set_ylabel('Lifetime maximum intensity (knots)')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{basin_code}: MHW baseline 1992–2021')
    ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top')


fig.delaxes(axes[5])
plt.tight_layout()
output_file = "mhw_plot/FigS12.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])

def count_indices_by_year(index_list, start_year=1982, end_year=2024):
    years = [int(str(idx)[-4:]) for idx in index_list]
    year_counts = Counter(years)
    result = {}
    for year in range(start_year, end_year + 1):
        result[year] = year_counts.get(year, 0)
    return result

years = list(range(1982, 2024))

fig, axes = plt.subplots(1, 2, figsize=(7, 4.5))
axes = axes.flatten()

num_wtmhw_b8211 = count_indices_by_year(name_wtmhw_b8211['name'].unique())
num_nomhw_b8211 = count_indices_by_year(name_nomhw_b8211['name'].unique())
wtmhw_values_b8211 = [num_wtmhw_b8211[year] for year in years]
nomhw_values_b8211 = [num_nomhw_b8211[year] for year in years]
slope_wtmhw, intercept_wtmhw, _, _, _ = stats.linregress(years, wtmhw_values_b8211)
print(stats.linregress(years, wtmhw_values_b8211))
trend_wtmhw = [slope_wtmhw * year + intercept_wtmhw for year in years]
slope_nomhw, intercept_nomhw, _, _, _ = stats.linregress(years, nomhw_values_b8211)
print(stats.linregress(years, nomhw_values_b8211))
trend_nomhw = [slope_nomhw * year + intercept_nomhw for year in years]
axes[0].plot(years, wtmhw_values_b8211, color='red',  marker='o', markersize=4, label='With-MHW', linewidth=1.5)
axes[0].plot(years, nomhw_values_b8211, color='blue', marker='o', markersize=4, label='No-MHW',   linewidth=1.5)
axes[0].plot(years, trend_wtmhw, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[0].plot(years, trend_nomhw, color='blue', linestyle='--', linewidth=2, alpha=0.7)
axes[0].set_ylabel('TC number')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)
axes[0].set_title('GL: MHW baseline 1982–2011')
axes[0].text(0.02, 0.98, 'a', transform=axes[0].transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top')

num_wtmhw_b9221 = count_indices_by_year(name_wtmhw_b9221['name'].unique())
num_nomhw_b9221 = count_indices_by_year(name_nomhw_b9221['name'].unique())
wtmhw_values_b9221 = [num_wtmhw_b9221[year] for year in years]
nomhw_values_b9221 = [num_nomhw_b9221[year] for year in years]
slope_wtmhw, intercept_wtmhw, _, _, _ = stats.linregress(years, wtmhw_values_b9221)
print(stats.linregress(years, wtmhw_values_b9221))
trend_wtmhw = [slope_wtmhw * year + intercept_wtmhw for year in years]
slope_nomhw, intercept_nomhw, _, _, _ = stats.linregress(years, nomhw_values_b9221)
print(stats.linregress(years, nomhw_values_b9221))
trend_nomhw = [slope_nomhw * year + intercept_nomhw for year in years]
axes[1].plot(years, wtmhw_values_b9221, color='red',  marker='o', markersize=4, label='With-MHW', linewidth=1.5)
axes[1].plot(years, nomhw_values_b9221, color='blue', marker='o', markersize=4, label='No-MHW',   linewidth=1.5)
axes[1].plot(years, trend_wtmhw, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[1].plot(years, trend_nomhw, color='blue', linestyle='--', linewidth=2, alpha=0.7)
axes[1].set_ylabel('TC number')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)
axes[1].set_title('GL: MHW baseline 1992–2021')
axes[1].text(0.02, 0.98, 'a', transform=axes[1].transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top')

plt.tight_layout()
output_file = "mhw_plot/Fig6.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])

fig, axes = plt.subplots(2, 3, figsize=(10, 9))
axes = axes.flatten()

for idx, (basin_code, basin_name, filter_func) in enumerate(basins):
    ax = axes[idx]
    basin_wtmhw = name_wtmhw_b8211[filter_func(name_wtmhw_b8211['name'])]
    basin_nomhw = name_nomhw_b8211[filter_func(name_nomhw_b8211['name'])]
    num_wtmhw = count_indices_by_year(basin_wtmhw['name'].unique())
    num_nomhw = count_indices_by_year(basin_nomhw['name'].unique())
    wtmhw_values = [num_wtmhw[year] for year in years]
    nomhw_values = [num_nomhw[year] for year in years]
    slope_wtmhw, intercept_wtmhw, _, _, _ = stats.linregress(years, wtmhw_values)
    print(stats.linregress(years, wtmhw_values))
    trend_wtmhw = [slope_wtmhw * year + intercept_wtmhw for year in years]
    slope_nomhw, intercept_nomhw, _, _, _ = stats.linregress(years, nomhw_values)
    print(stats.linregress(years, nomhw_values))
    trend_nomhw = [slope_nomhw * year + intercept_nomhw for year in years]
    ax.plot(years, wtmhw_values, color='red',  marker='o', markersize=4, label='With-MHW', linewidth=1.5)
    ax.plot(years, nomhw_values, color='blue', marker='o', markersize=4, label='No-MHW',   linewidth=1.5)
    ax.plot(years, trend_wtmhw, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.plot(years, trend_nomhw, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_ylabel('TC number')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{basin_code}: MHW baseline 1982–2011')
    ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top')


fig.delaxes(axes[5])
plt.tight_layout()
output_file = "mhw_plot/FigS13.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])

fig, axes = plt.subplots(2, 3, figsize=(10, 9))
axes = axes.flatten()

for idx, (basin_code, basin_name, filter_func) in enumerate(basins):
    ax = axes[idx]
    basin_wtmhw = name_wtmhw_b9221[filter_func(name_wtmhw_b9221['name'])]
    basin_nomhw = name_nomhw_b9221[filter_func(name_nomhw_b9221['name'])] 
    num_wtmhw = count_indices_by_year(basin_wtmhw['name'].unique())
    num_nomhw = count_indices_by_year(basin_nomhw['name'].unique())
    wtmhw_values = [num_wtmhw[year] for year in years]
    nomhw_values = [num_nomhw[year] for year in years]
    slope_wtmhw, intercept_wtmhw, _, _, _ = stats.linregress(years, wtmhw_values)
    print(stats.linregress(years, wtmhw_values))
    trend_wtmhw = [slope_wtmhw * year + intercept_wtmhw for year in years]
    slope_nomhw, intercept_nomhw, _, _, _ = stats.linregress(years, nomhw_values)
    print(stats.linregress(years, nomhw_values))
    trend_nomhw = [slope_nomhw * year + intercept_nomhw for year in years]
    ax.plot(years, wtmhw_values, color='red',  marker='o', markersize=4, label='With-MHW', linewidth=1.5)
    ax.plot(years, nomhw_values, color='blue', marker='o', markersize=4, label='No-MHW',   linewidth=1.5)
    ax.plot(years, trend_wtmhw, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.plot(years, trend_nomhw, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_ylabel('TC number')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{basin_code}: MHW baseline 1992–2021')
    ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top')


fig.delaxes(axes[5])
plt.tight_layout()
output_file = "mhw_plot/FigS14.pdf"
plt.savefig(output_file)
plt.close()
subprocess.run(['open', output_file])
