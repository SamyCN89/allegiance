#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:26:30 2024

@author: samy
"""

#%%
from calendar import c
from matplotlib import pyplot as plt
import numpy as np
import time
# from functions_analysis import *
from pathlib import Path
import sys
sys.path.append('../../shared_code')

from fun_loaddata import *
from fun_dfcspeed import *

from fun_metaconnectivity import (compute_metaconnectivity, 
                                  intramodule_indices_mask, 
                                  get_fc_mc_indices, 
                                  get_mc_region_identities, 
                                  fun_allegiance_communities,
                                  compute_trimers_identity,
                                    build_trimer_mask,
                                  )

from fun_utils import (set_figure_params, 
                       get_paths, 
                       load_cognitive_data,
                       load_timeseries_data,
                       load_grouping_data,
                       )
# =============================================================================
# This code compute 
# Load the data
# Intersect the 2 and 4 months to have data that have the two datapoints
# ========================== Figure parameters ================================
save_fig = set_figure_params(False)

# =================== Paths and folders =======================================
timeseries_folder = 'Timecourses_updated_03052024'
external_disk = True
if external_disk==True:
    root = Path('/media/samy/Elements1/Proyectos/LauraHarsan/script_mc/')
else:    
    root = Path('/home/samy/Bureau/Proyect/LauraHarsan/Ines/')

paths = get_paths(external_disk=True,
                  external_path=root,
                  timecourse_folder=timeseries_folder)

# ========================== Load data =========================
cog_data_filtered = load_cognitive_data(paths['sorted'] / 'cog_data_sorted_2m4m.csv')
data_ts = load_timeseries_data(paths['sorted'] / 'ts_and_meta_2m4m.npz')
mask_groups, label_variables = load_grouping_data(paths['results'] / "grouping_data_oip.pkl")


# ========================== Indices ==========================================
ts=data_ts['ts']
n_animals = data_ts['n_animals']
regions = data_ts['regions']
anat_labels = data_ts['anat_labels']

#Binarize the ts of one animal
ts = np.array([np.where(ts[i] > np.std(ts[i])/10, 1, 0) for i in range(n_animals)])


# Plot the ts of one animal in a matrix
plt.figure(figsize=(10, 5))
plt.imshow(ts[0].T, aspect='auto', interpolation='none', cmap='Greys')
plt.colorbar()
plt.title('Time series of one animal')
plt.xlabel('Time')
plt.ylabel('Regions')
if save_fig:
    plt.savefig(paths['figures'] / 'ts_animal_0.png', dpi=300, bbox_inches='tight')
plt.show()
# plt.figure(figsize=(10, 5))
# plt.imshow(ts[0], aspect='auto')
# plt.colorbar()
# plt.title('Time series of one animal')
# %%
#Plot the ts of one animal in plot
plt.figure(figsize=(10, 5))
plt.clf()
plt.plot(ts[0], color='grey', alpha=0.1)
plt.title('Time series of one animal')
plt.xlabel('Time')
plt.ylabel('Regions')
# plt.xlim(0, 1000)
plt.ylim(0, 1)
plt.show()
