import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
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
df = df[0:100]

ds = xr.open_dataset('~/mw/1998_2024-REMSS-L4_GHRSST-SSTfnd-MW_OI-GLOB-v02.0-fv05.1.nc')
da = ds['analysed_sst']

values = da.sel(time=df['time'].apply(lambda x: x.replace(hour=12, minute=0, second=0, microsecond=0)).values,
                lon=df['grid_lon'].values, lat=df['grid_lat'].values).values

print(values.shape)
