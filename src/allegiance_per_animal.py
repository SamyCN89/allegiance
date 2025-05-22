#%%
from matplotlib import cm
import numpy as np
from pathlib import Path
from shared_code.fun_utils import get_paths
from torch import alpha_dropout
from tqdm import tqdm
import matplotlib.pyplot as plt
from shared_code.fun_metaconnectivity import load_merged_allegiance

# Set consistent config to match previous run
window_size = 9
lag = 1
timecourse_folder = 'Timecourses_updated_03052024'

# Load meta info to determine shape
paths = get_paths(timecourse_folder=timecourse_folder)
#%%
data_ts = np.load(paths['sorted'] / 'ts_and_meta_2m4m.npz', allow_pickle=True)
ts = data_ts['ts']
n_animals = len(ts)
n_regions = ts[0].shape[1]

filename_dfc = f'window_size={window_size}_lag={lag}_animals={n_animals}_regions={n_regions}'
dfc_data = np.load(paths['mc'] / f'dfc_{filename_dfc}.npz')
n_windows = np.transpose(dfc_data['dfc_stream'], (0, 3, 2, 1)).shape[1]



#%%

#Load the merged allegiance data
dfc_communities, sort_allegiances, contingency_matrices = load_merged_allegiance(paths, window_size=9, lag=1)

#%%
dfc_communities_sorted = dfc_communities[0, :, sort_allegiances[0, 0].astype(int)]
# Plot the mean matrix
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Allegiance Matrix - Animal 0, Window 0")
# plt.imshow(cm_0_mean.T , aspect='auto', interpolation='none', cmap='Greys')
plt.imshow(dfc_communities_sorted , aspect='auto', interpolation='none', cmap='Greys')
# plt.clim(0, 1)
plt.colorbar()
plt.xlabel("DFT Frequency")
plt.ylabel("DFT Frequency")
plt.show()

#%%%# Check the shape of the loaded data

cm_0 = np.array(contingency_matrices[0])

triu_indices = np.array(np.triu_indices(n_regions, k=1))


cm_0_triu = cm_0[:, triu_indices[0], triu_indices[1]]


cm_0_mean_triu = np.mean(cm_0_triu, axis=0) 

#Reshape the mean matrix to the original shape
cm_0_mean = np.zeros((n_regions, n_regions))
cm_0_mean[triu_indices[0], triu_indices[1]] = cm_0_mean_triu
cm_0_mean = cm_0_mean + cm_0_mean.T
cm_0_mean[np.diag_indices_from(cm_0_mean)] = 1

# Plot the mean matrix
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Allegiance Matrix - Animal 0, Window 0")
plt.imshow(cm_0_mean.T , aspect='auto', interpolation='none', cmap='Greys')
# plt.imshow(dfc_communities_sorted , aspect='auto', interpolation='none', cmap='Greys')
# plt.clim(0, 1)
plt.colorbar()
plt.xlabel("DFT Frequency")
plt.ylabel("DFT Frequency")
plt.show()


#%%
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
# %%

# Global Allegiance Matrix

# Average modular structure over all windows and animals
allegiance_matrices = cm_0
allegiance_avg = allegiance_matrices.mean(axis=0)

# "consensus" community structure over the whole period with Louvain method
from shared_code.fun_metaconnectivity import contingency_matrix_fun
allegancy_communities, argsort_allegancy_communities, allegancy_modularity_q, contingency_matrix =contingency_matrix_fun(1000, mc_data=allegiance_avg, gamma_range=10, gmin=0.1, gmax=1, cache_path=None, ref_name='', n_jobs=-1)
#%%
consensus_community =contingency_matrix

#Plot consensus community
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Allegiance Matrix - Animal 0, Window 0")
plt.imshow(consensus_community , aspect='auto', interpolation='none', cmap='Greys')
plt.colorbar()
plt.xlabel("DFT Frequency")
plt.ylabel("DFT Frequency")
plt.show()
# Plot the mean matrix
# %%
from sklearn.cluster import KMeans

# Reshape to (n_windows, n_nodes*n_nodes)
X = allegiance_matrices.reshape(allegiance_matrices.shape[0], -1)

kmeans = KMeans(n_clusters=3, random_state=42)
state_labels = kmeans.fit_predict(X)
# %%
from sklearn.metrics.pairwise import cosine_distances

D = cosine_distances(X) 
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Allegiance Matrix - Animal 0, Window 0")
plt.imshow(D , aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()
plt.xlabel("DFT Frequency")
plt.ylabel("DFT Frequency")
plt.show()
# %%
import matplotlib.pyplot as plt

plt.plot(state_labels, marker='o')
plt.xlabel("Time window")
plt.ylabel("Modular state")
plt.title("Modular state dynamics")
plt.grid(True)
plt.show()

# %%
