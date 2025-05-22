#%%
from matplotlib import cm
import numpy as np
from pathlib import Path
from shared_code.fun_utils import get_paths
from torch import alpha_dropout


# Set consistent config to match previous run
window_size = 9
lag = 1
timecourse_folder = 'Timecourses_updated_03052024'

# Load meta info to determine shape
paths = get_paths(timecourse_folder=timecourse_folder)
data_ts = np.load(paths['sorted'] / 'ts_and_meta_2m4m.npz', allow_pickle=True)
ts = data_ts['ts']
n_animals = len(ts)
n_regions = ts[0].shape[1]

filename_dfc = f'window_size={window_size}_lag={lag}_animals={n_animals}_regions={n_regions}'
dfc_data = np.load(paths['mc'] / f'dfc_{filename_dfc}.npz')
n_windows = np.transpose(dfc_data['dfc_stream'], (0, 3, 2, 1)).shape[1]

# Define output and temp directory
out_dir = paths['allegiance'] / 'temp'
merged_out_file = paths['allegiance'] / f'merged_allegiance_{filename_dfc}.npz'

#%%
# Preallocate data structures (list-based to be safe with variable shapes)
dfc_communities = [[None for _ in range(n_windows)] for _ in range(n_animals)]
sort_allegiances = [[None for _ in range(n_windows)] for _ in range(n_animals)]
contingency_matrices = [[None for _ in range(n_windows)] for _ in range(n_animals)]

#%%
# Load each file and insert into arrays
a = 0
for w in range(n_windows):
    out_file = out_dir / f"{filename_dfc}_animal_{a:02d}_window_{w:04d}.npz"
    if out_file.exists():
        data = np.load(out_file)
        dfc_communities[a][w] = data["dfc_communities"]
        sort_allegiances[a][w] = data["sort_allegiance"]
        contingency_matrices[a][w] = data["contingency_matrix"]
    else:
        print(f"[MISSING] Animal {a}, Window {w}")
#%%
# Save the full structure
np.savez_compressed(
    merged_out_file,
    dfc_communities=dfc_communities,
    sort_allegiances=sort_allegiances,
    contingency_matrices=contingency_matrices
)
print(f"[DONE] Merged data saved to: {merged_out_file}")

# %%


import matplotlib.pyplot as plt

#%%%# Check the shape of the loaded data

cm_0 = np.array(contingency_matrices[0])

triu_indices = np.array(np.triu_indices(n_regions, k=1))


cm_0_triu = cm_0[:, triu_indices[0], triu_indices[1]]

# Plot matrices 9 matrices in a grid
plt.figure(figsize=(10, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.clf()
    plt.title(f"Contingency Matrix - Animal 0, Window {i}")
    plt.imshow(cm_0[i].T , aspect='auto', interpolation='none', cmap='Greys')
    plt.clim(0, 1)
    plt.colorbar()
    plt.xlabel("DFT Frequency")
    plt.ylabel("DFT Frequency")
#%%
# plot imshow cm_0_triu
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Contingency Matrix - Animal 0, Window 0")
plt.imshow(cm_0_triu.T , aspect='auto', interpolation='none', cmap='Greys')
plt.clim(0, 1)
plt.colorbar()
plt.xlabel("DFT Frequency")
plt.ylabel("DFT Frequency")
plt.show()

#%%
# Plot the cumsum of cm_0_triu contingency matrix for all the windows
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Contingency Matrix - Animal 0, Window 0")
plt.plot(np.sort(cm_0_triu.ravel()))# aspect='auto', interpolation='none', cmap='Greys')
#%%
# Plot the histogram of the contingency matrix for all windows
plt.figure(figsize=(12, 12))
plt.title("Contingency Matrix - Animal 0, Window 0")

# One histogram per row (i.e., each region pair)
plt.hist(cm_0_triu[cm_0_triu > 0.1], bins=100, density=True, histtype='step')
plt.xlabel("Contingency Matrix Value")
plt.ylabel("Frequency") 
plt.ylim(0, 2)
plt.tight_layout()
plt.show()
# %%
# Plot imshow of the contingency matrix for  9 windows in one animal
plt.figure(figsize=(12, 12))
plt.clf()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.title(f"Contingency Matrix - Animal 0, Window {i}")
    plt.imshow(cm_0[i].T , aspect='auto', interpolation='none', cmap='viridis')
    plt.clim(0, 1)
    plt.colorbar()
    plt.xlabel("DFT Frequency")
    plt.ylabel("DFT Frequency")