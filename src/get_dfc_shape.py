from shared_code.fun_utils import get_paths, load_timeseries_data
import numpy as np

paths = get_paths(timecourse_folder='Timecourses_updated_03052024')
data_ts = load_timeseries_data(paths['sorted'] / 'ts_and_meta_2m4m.npz')
ts = data_ts['ts']
n_animals = len(ts)
n_regions = ts[0].shape[1]

# Load precomputed DFC stream
dfc_path = paths['mc'] / f'dfc_window_size=9_lag=1_animals={n_animals}_regions={n_regions}.npz'
dfc = np.load(dfc_path)
dfc_stream = dfc['dfc_stream']
n_windows = dfc_stream.shape[-1]

print(f"{n_animals} {n_windows}")
