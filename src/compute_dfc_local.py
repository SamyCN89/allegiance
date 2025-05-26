#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:26:30 2024

@author: samy
"""

#%%
from calendar import c
from matplotlib import pyplot as pltcomputation
import numpy as np
import time
from pathlib import Path

# from sphinx import ret
from shared_code.fun_loaddata import *
from shared_code.fun_dfcspeed import *
from shared_code.fun_metaconnectivity import *


from shared_code.fun_utils import (set_figure_params, 
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

#%%

# =================== Paths and folders =======================================
# Will prioritize PROJECT_DATA_ROOT if set
timeseries_folder = 'Timecourses_updated_03052024'
paths = get_paths(timecourse_folder=timeseries_folder)


# ========================== Load data =========================
cog_data_filtered = load_cognitive_data(paths['sorted'] / 'cog_data_sorted_2m4m.csv')
data_ts = load_timeseries_data(paths['sorted'] / 'ts_and_meta_2m4m.npz')
mask_groups, label_variables = load_grouping_data(paths['sorted'] / "grouping_data_oip.pkl")


# ========================== Indices ==========================================
ts=data_ts['ts']
n_animals = data_ts['n_animals']
regions = data_ts['regions']
anat_labels = data_ts['anat_labels']



#%% Compute the DFC stream
#Parameters speed

PROCESSORS =-1

lag=1
tau=5
window_size = 9
window_parameter = (5,100,1)

#Parameters allegiance analysis
n_runs_allegiance = 1000
gamma_pt_allegiance = 100

tau_array       = np.append(np.arange(0,tau), tau ) 
lentau          = len(tau_array)

time_window_min, time_window_max, time_window_step = window_parameter
time_window_range = np.arange(time_window_min,
                              time_window_max+1,
                              time_window_step)


#%%
#%%compute dfc stream
# Compute the DFC stream
# Define the wrapper function
# def compute_for_window_size(ws):
#     print(f"Starting DFC computation for window_size={ws}")
#     start = time.time()
#     dfc_stream = compute_dfc_stream(
#         ts,
#         window_size=ws,
#         lag=lag,
#         n_jobs=1,  # Important: Set to 1 to avoid nested parallelism
#         save_path=paths[prefix],
#     )
#     stop = time.time()
#     print(f"Finished window_size={ws} in {stop - start:.2f} sec")
#     # return ws, dfc_stream

# #%%load_npz_cache
# # #test compute_for_window_size
# # from shared_code.fun_dfcspeed import *
# #Uncomment to test the function for a specific window size
# # ws, dfc_stream = compute_for_window_size_new(101)
# # ws2, dfc_stream2 = compute_for_window_size(101)

# #%%
# # Run parallel dfc stream over window sizes 
# start = time.time()
# Parallel(n_jobs=min(PROCESSORS, len(time_window_range)))(
#     delayed(compute_for_window_size)(ws) for ws in time_window_range
# )

# stop = time.time()
# print(f'DFC stream computation time {stop-start}')

# %%
# Check for missing DFC stream files and compute if necessary function

    # return ws, dfc_stream

prefix='dfc'

def get_tnet_window_range(time_window_range, prefix='dfc'):
    """
    Get the range of window sizes for tnet files.
    Args:
        prefix (str): Prefix for the tnet files.
    Returns:
        list: List of window sizes.
    """
    def compute_for_window_size_new(ws,prefix='dfc'):
        print(f"Starting DFC computation for window_size={ws}")
        start = time.time()
        dfc_stream = handler_tnet_analysis(
            ts,
            prefix=prefix,
            window_size=ws,
            lag=lag,
            n_jobs=1,  # Important: Set to 1 to avoid nested parallelism
            save_path=paths[prefix],
        )
        stop = time.time()
        print(f"Finished window_size={ws} in {stop - start:.2f} sec")

    # Run the check and complete function   
    missing_files = check_and_rerun_missing_files(
        paths[prefix], prefix, time_window_range, lag, n_animals, regions
    )

    if missing_files:
        time_window_range = np.array(missing_files)

    # Run parallel dfc stream over window sizes 
    start = time.time()
    Parallel(n_jobs=min(PROCESSORS, len(time_window_range)))(
        delayed(compute_for_window_size_new)(ws, prefix) for ws in time_window_range
    )

    stop = time.time()
    print(f'{prefix} stream computation time {stop-start}')

    # Check for missing prefix files and compute if necessary function
    missing_files = check_and_rerun_missing_files(
        paths[prefix], prefix, time_window_range, lag, n_animals, regions
    )
get_tnet_window_range(time_window_range, prefix='dfc')
# %%
