from pathlib import Path
import numpy as np
from sst_loader import load_windows

data, name = load_windows(Path('t_data'), with_date=False, with_name=True, only_TSHU_status=False)
name.to_pickle('name.pkl')
np.save('t_data/oisst.npy', data)

