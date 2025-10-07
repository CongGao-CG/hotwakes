import math
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sst_loader import load_windows

_, name = load_windows(Path('t_data'), with_date=False, with_name=True, only_TSHU_status=False)
name = name.reset_index(drop=True)
name = name[['name', 'lon', 'lat', 'time']]
name['grid_lat'] = round((name['lat'] - 0.125) / 0.25) * 0.25 + 0.125
name['grid_lon'] = round((name['lon'] - 0.125) / 0.25) * 0.25 + 0.125
name.loc[name['grid_lon'] == 180.125, 'grid_lon'] = 179.785

name['btime'] = name.groupby('name')['time'].transform('min')
name = name[name['btime'] >= '1998-01-16 00:00:00']

df = name.loc[name.index.repeat(31)].reset_index(drop=True)
offsets = np.tile(np.arange(-15, 16), len(name))
df['time'] = df['time'] + pd.to_timedelta(offsets, unit='D')
df['grid_time'] = df['time'].dt.normalize() + pd.Timedelta(hours=12)

ds = xr.open_dataset('~/Princeton Dropbox/Cong Gao/Downloads/mw/1998_2024-REMSS-L4_GHRSST-SSTfnd-MW_OI-GLOB-v02.0-fv05.1.nc')
# ds = xr.open_dataset('~/mw/1998_2024-REMSS-L4_GHRSST-SSTfnd-MW_OI-GLOB-v02.0-fv05.1.nc')
da = ds['analysed_sst']

# Option 1
'''
values = np.empty(df.shape[0])
for i, (idx, row) in enumerate(df.iterrows()):
    values[i] = da.sel(time=row['grid_time'],
                       lon=row['grid_lon'],
                       lat=row['grid_lat']).values
'''
# Option 2 (not worked for one-time selection and too slow)
'''
values = np.array([])
ll = 500_000
n_max = math.floor(df.shape[0]/ll)
for n in np.arange(n_max+1):
    if n == n_max:
        df_temp = df[(n*ll):]
    else:
        df_temp = df[(n*ll):(ll*n+ll)]
    v_temp = da.sel(time=xr.DataArray(df_temp['grid_time'].values, dims='points'),
                    lon=xr.DataArray(df_temp['grid_lon'].values, dims='points'),
                    lat=xr.DataArray(df_temp['grid_lat'].values, dims='points')).values
    values = np.concatenate([values, v_temp])
'''
# Option 3
def extract_value(row):
    return da.sel(
        time=row['grid_time'],
        lon=row['grid_lon'],
        lat=row['grid_lat']
    ).values
values = Parallel(n_jobs=-1)(
    delayed(extract_value)(row) 
    for _, row in df.iterrows()
)
values = np.array(values)

np.save('t_data/mw_501.npy', values)
