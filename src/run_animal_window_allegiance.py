
#%%
from fileinput import filename
import sys
from isort import file
import numpy as np
from pathlib import Path
from shared_code.fun_utils import get_paths, load_timeseries_data
from shared_code.fun_metaconnectivity import fun_allegiance_communities

#%%
# Read arguments
task_id = int(sys.argv[1])  # SLURM_ARRAY_TASK_ID
n_animals = int(sys.argv[2])
n_windows = int(sys.argv[3])

# Compute animal + window index
animal_idx = task_id // n_windows
window_idx = task_id % n_windows

# Parameters
n_runs = 1000
gamma_pt = 100
processors = 8
window_size = 9
lag = 1

# Get paths
paths = get_paths(timecourse_folder='Timecourses_updated_03052024')
data_ts = load_timeseries_data(paths['sorted'] / 'ts_and_meta_2m4m.npz')
ts = data_ts['ts']

# make the name of the animal and window index
filename_dfc = f'window_size={window_size}_lag={lag}_animals={len(ts)}_regions={ts[0].shape[1]}'

# Load DFC
dfc_data = np.load(paths['mc'] / f'dfc_{filename_dfc}.npz')
dfc_stream = np.transpose(dfc_data['dfc_stream'], (0, 3, 2, 1))  # shape: (n_animals, n_windows, n_regions, n_regions)

# Output path
out_path = paths['allegiance'] / 'temp' / f'{filename_dfc}_animal_{animal_idx:02d}_window_{window_idx:04d}.npz'
out_path.parent.mkdir(parents=True, exist_ok=True)

# Skip if already done
if out_path.exists():
    print(f"[SKIP] Animal {animal_idx}, Window {window_idx} already exists.")
    sys.exit(0)

print(f"[RUN] Animal {animal_idx}, Window {window_idx}")

# Compute allegiance
dfc = dfc_stream[animal_idx, window_idx]
dfc_com, sort_all, cont_mat = fun_allegiance_communities(
    dfc,
    n_runs=n_runs,
    gamma_pt=gamma_pt,
    save_path=None,
    ref_name=f'animal_{animal_idx:02d}_window_{window_idx:04d}',
    n_jobs=processors
)

np.savez_compressed(out_path,
                    dfc_communities=dfc_com,
                    sort_allegiance=sort_all,
                    contingency_matrix=cont_mat)
print(f"[DONE] Animal {animal_idx}, Window {window_idx} saved to {out_path}")