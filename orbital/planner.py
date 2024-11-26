"""
This should contain routines for graph-based and standard
    cost-based traversals of a heuristic map.
"""
from sklearn.metrics.pairwise import haversine_distances
import numpy as np
# from skimage.filters import peak_local_max 

# Constants
WORST_CASE_SLEW_PER_ACTION = np.pi

# Class for handling everything relating to a particular host
class Observer:
    def __init__(self) -> None:
        pass # Should hold a planner object with all potential and actual plans represented, as well as all balltrees, LUTs, observer-centric methods, and relative targeting methods

# Class to hold all of the plans and targeting metadata
class Planner:
    def __init__(self, slew_rate=1.0) -> None: # Slew rate is radians/sec
        pass # Should hold plans, obs_info, and slew_info

# first point in any plan should be the last known observation point for continuity
def calc_path_length_haversine(plans):
    return np.sum(haversine_distances(plans[:-1], plans[1:]), -1)

# Plan should be sequential observation ra,dec pairs, optionally stored with many planned paths
def normalized_total_slew(planner):
    # Worst case should be considered slewing by pi (reversing your pointing) * the number of actions (obs)
    # obs in format (RA, Dec) of shape Nobs x 2 x N_plans
    total_dist = calc_path_length_haversine(planner.obs_plans)

    # If multiple plans, multiple corresponding distances are calculated (per plan)
    return total_dist / (planner.num_obs_per_plan * WORST_CASE_SLEW_PER_ACTION)

def plan_efficiency(slew_info, obs_info):
    return obs_info["elapsed_t"] / (slew_info["elapsed_t"] + obs_info["elapsed_t"])

# obs is all query pointings, balltrees allow lookup at runtime of specific target info, max_angrelvel is FOV/dwell so it's nice and normalized, min_vmag is our minimum detectable quality observation 'brightness' for any target
def calculate_obs_quality(obs, balltrees, max_angrelvel=1/3, min_vmag=12.): # Arbitrary default values
    return

# RH: TODO - Make sure all values and costs are on a comparable scale???

# Value/Reward function for single time-step given approximate density model
def calc_value(obs): # Doesn't care about initial position
    return

#                                   unitless                          sum(det(cov(sat_i))) for all i     *OR*    staleness_index
# Basically this is: cost = normalized_total_slew() / plan_efficiency() * (total_covariance_volume  |  staleness_index)
# Single timestep, stemming from previous position
def calc_cost(init_obs, obs):
    return

# arr is treated as a set of 2D arrays stored depth-wise (last dimension)
# def find_local_extrema(arr):
#     if arr.ndim == 3:
#         inds = []
#         for i in range(arr.shape[2]):
#             inds.append(peak_local_max(arr[:,:,i]))
#     else:
#         inds = peak_local_max(arr)

#     return inds

# find global maximum value indices
def global_argmax(arr):
    return np.where(arr == arr.max())

# GREEDY OBS PLANNER
def greedy_obs_plan_gen(observers, time_window_start, time_window_end):
    # For each action, FOR EACH SENSOR, choose the highest value, lowest cost observation to make. Uses global armax with associated values and costs and then lexicographically sorts by value then by cost
    observers = observers.copy()
    while observers:
        observers = sorted(observers) # __lt__ should be used to compare two observers by latest_timestamp (end of previous observation), with earliest going first
        keep_list = []
        for i,o in enumerate(observers):
            # get highest value indices

            # get their associated costs

            # lexicographical sort

            # Log our choice (or choices...) in o.planner via appending the vector of obs and obs_info, slew_info

            # Update observer last_observation_end

            # Update observation-dependent, time-varying target states (last observation time, new covariance, ...)

            # Finally add to keep list
            if o.last_obs_end < time_window_end:     
                keep_list.append(o)
        observers = keep_list

    # Repeat until all actions selected

    return