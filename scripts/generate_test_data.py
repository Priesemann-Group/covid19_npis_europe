import sys
import logging
import time
import os

sys.path.append("../")

import covid19_npis
import numpy as np

params = {
    # population size per country and age group
    "N": np.array([[1e15, 1e15, 1e15, 1e15], [1e15, 1e15, 1e15, 1e15]]),
    # Reproduction number at t=0 per country and age group
    "R_0": np.array([[2.31, 2.32, 2.33, 2.34], [2.31, 2.32, 2.33, 2.34]]),
    # Initial infected
    "I_0": np.array([[10, 10, 10, 10], [10, 10, 10, 10]], dtype="float64"),
    # Change point date/index
    "d_cp": np.array([[15, 16, 18, 20], [15, 16, 18, 20]]),
    # Length of the change point
    "l_cp": 5.2,
    # Alpha value of the change point
    "alpha_cp": np.array([[0.73, 0.72, 0.74, 0.75], [0.73, 0.72, 0.74, 0.75]]),
    # Number of timesteps
    "t_max": 80,
    # Number of Nans before data, overall length is nans+tmax
    "num_nans": 10,
    # Generation interval
    "g": 4.0,
}

# Create test country dataset with params above
# One intervention and changepoint
covid19_npis.test_data.data_generators.save_data("../data", **params)
