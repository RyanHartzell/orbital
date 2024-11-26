"""
This uses all our wonderful access and density calculations to create a greedy plan
"""
from density import *
from access import in_major_keep_out_zones, not_sunlit, out_of_range
from sklearn.metrics.pairwise import haversine_distances
import numpy as np
# from skimage.filters import peak_local_max 
from datetime import timedelta
from astropy import units as u

# Constants
WORST_CASE_SLEW_PER_ACTION = np.pi

class Observer:
    def __init__(self, host, host_ind):
        self.host = host
        self.host_ind = host_ind
        self.last_observation_end_time = None
        self.plan = [] # Contains flat indices into RA/DEC meshgrid (from density module)
        self.reward = [] # Size of plan-1, should be all associated rewards for actions
        self.cost = [] # Size of plan-1, should be all associated costs for actions

    def __lt__(self, other):
        return self.last_observation_end_time < other.last_observation_end_time

    def as_dict(self):
        return {
            "Index": self.host_ind,
            "NoradID": self.host.ID,
            "Plan": self.plan,
            "Rewards": self.reward,
            "TotalReward": np.sum(self.reward),
            "Costs": self.cost,
            "TotalCost": np.sum(self.cost)
        }

    def save(self, fname):
        # This should save to disk whatever Observer data we want as a pandas dataframe CSV maybe?
        with open(fname, 'w') as f:
            f.write(json.dumps())

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
    return np.where(np.isclose(arr, arr.max()))

# GREEDY OBS PLANNER
def greedy_obs_plan_gen(observers, time_window_start, time_window_end):
    # For each action, FOR EACH SENSOR, choose the highest value, lowest cost observation to make. Uses global armax with associated values and costs and then lexicographically sorts by value then by cost
    observers = observers.copy()
    final = []

    # Get initial states for each observer (just straight up best move as of time_window_start)
    for o in observers:
        o.last_observation_end_time = time_window_start
        init_greedy(o, targets)

    # Init target records
    target_records = [{"last_seen": time_window_start, "last_uncertainty": 1.0} for _ in targets]

    while observers:
        observers = sorted(observers) # __lt__ should be used to compare two observers by latest_timestamp (end of previous observation), with earliest going first
        for i,o in enumerate(observers):
            # do execute greedy step

            # Observer data should be updated at the end of this (reward, cost, new state, last observation end time)
            # TargetRecords should be updated at the end of this function as well (last seen time, last uncertainty)
            execute_greedy_step(o, targets, target_records)

            if o.last_obs_end > time_window_end:     
                final.append(observers.pop(i))

    # All observation chains are stored on the objects returned in final (which may be in a differnt order than the initial observer list)
    return final

# Uncertainty update (U0 is just km, dt is expected to be a timedelta object)
def update_uncertainty(U0, dt, rate=0.1/3600): # rate is 0.1 km/h converted to km/s 
    return U0 + rate*U0*dt.total_seconds()

# Integration time + slew time
# def obs_duration(slew, avg_target_distance):
def obs_duration(slew, slew_rate=np.pi/4, frames=7, integration=0.5):
    # For simplicity (slew in radians / slew rate in rad/s) + (frames unitless * integration in s) = duration in s
    return slew / slew_rate + frames * integration

# Value (observation/collection quality)
# Target records should just be a list of the same length as the full targets list, but with a dictionary of data stored at each element (essentially per target)
def compute_reward(t, target_records, access, query_result, max_allowable_unseen=timedelta(hours=1.0)):
    # Get subset of target record dictionaries
    targ_recs = target_records[access][query_result] # Query result is flattened index w.r.t accessible targets

    # Compute dt = t - target_records.last_seen
    # Compute the absolute magnitude uncertainty reduction for all targets
    # Compute the absolute magnitude change in staleness index for all targets
    reward = 0.0
    for tr in targ_recs:
        dt = t - tr["last_seen"] # Timedelta, which I believe is evaluated as seconds

        # 0.1 km is the default reset value for uncertainty
        reward += (update_uncertainty(tr["last_uncertainty"], dt) - 0.1 + (dt > max_allowable_unseen))

    return reward

# Cost (haversine_costs(a, b))
def compute_cost(current_state_index, new_state_index):
    return haversine_distances(np.c_[RA.flat[current_state_index],DEC.flat[current_state_index]], np.c_[RA.flat[new_state_index],DEC.flat[new_state_index]])

def init_greedy(o, targets):
    t = o.last_observation_end_time
    host = o.host

    # Get access mask
    sunlit_access = not_sunlit([t], targets)
    # print(f"% access [SUNLIT] = {np.sum(~sunlit_access)/sunlit_access.size * 100.}")

    range_access = out_of_range([t], host, targets)
    # print(f"% access [IN-RANGE] = {np.sum(~range_access)/range_access.size * 100.}")

    koz_access = in_major_keep_out_zones([t], host, targets)
    # print(f"% access [NOT-IN-KOZ] = {np.sum(~koz_access)/koz_access.size * 100.}")

    # Construct overall access mask (should be SATNUM x TIMESTEP)
    access = ~sunlit_access * ~range_access * ~koz_access # We can multiply these since any zero value should cause a switch to False
    # print(f"Total % access across timesteps = {np.sum(access)/access.size * 100.}")

    # # Calculate apparent ra, dec, ranges relative to host state at each time t
    obs = reformat_radecrange(calculate_apparent_radecrange(host, np.asarray(targets), [t], access), ragged=True)

    # Build all ball trees
    bt = construct_ball_tree(obs[0][0], obs[1][0])

    # Calculate new value map using explicit radius_query method (not KDE, since we want total value based on target indices)
    density_map, query_results = construct_fov_density_map(bt)

    # Pick best DENSITY!!! index (argmax)
    new_state_index = np.ravel_multi_index(global_argmax(density_map), density_map.shape)[0] # I can change this to a random sample instead...

    # Set new values on observer
    o.current_state_index = new_state_index

    # # state'
    # return new_state_index


# We just assume the slew is part of the observation since we already have the asumption that none of these targets are moving faster than a field of view during observation, and slew time is negligible
def execute_greedy_step(o, targets, target_records):
    t = o.last_observation_end_time
    host = o.host

    # Get access mask
    sunlit_access = not_sunlit([t], targets)
    # print(f"% access [SUNLIT] = {np.sum(~sunlit_access)/sunlit_access.size * 100.}")

    range_access = out_of_range([t], host, targets)
    # print(f"% access [IN-RANGE] = {np.sum(~range_access)/range_access.size * 100.}")

    koz_access = in_major_keep_out_zones([t], host, targets)
    # print(f"% access [NOT-IN-KOZ] = {np.sum(~koz_access)/koz_access.size * 100.}")

    # Construct overall access mask (should be SATNUM x TIMESTEP)
    access = ~sunlit_access * ~range_access * ~koz_access # We can multiply these since any zero value should cause a switch to False
    # print(f"Total % access across timesteps = {np.sum(access)/access.size * 100.}")

    # # Calculate apparent ra, dec, ranges relative to host state at each time t
    obs = reformat_radecrange(calculate_apparent_radecrange(host, np.asarray(targets), [t], access), ragged=True)

    # Build all ball trees
    bt = construct_ball_tree(obs[0][0], obs[1][0])

    # Calculate new value map using explicit radius_query method (not KDE, since we want total value based on target indices)
    density_map, query_results = construct_fov_density_map(bt)

    # For each mesh point, calculate obs value
    value_map = np.zeros(density_map.size)
    for i,q in enumerate(query_results):
        value_map[i]=compute_reward(t, target_records, access, q)
    value_map = value_map.reshape(density_map.shape)

    # Pick best value index (argmax)
    new_state_index = np.ravel_multi_index(global_argmax(value_map), value_map.shape)[0] # I can change this to a random sample instead...

    # Technically should be moving this up into the value_map builder above, but assuming slew is negligible for now...
    slew = compute_cost(o.current_state_index, new_state_index)
    duration = obs_duration(slew)
 
    # Now update all the stuff we need on the observer!
    o.current_state_index = new_state_index
    o.last_observation_end_time = t + duration

    # For each target, update our global target record with last seen time and updated uncertainty!!!
    for tr in target_records[access][query_results[new_state_index]]:
        tr["last_uncertainty"] = update_uncertainty(tr["last_uncertainty"], t - tr["last_seen"])
        tr["last_seen"] = o.last_observation_end_time
 
    # # state'
    # return new_state_index

if __name__=="__main__":
    sats = load_satellites()

    # Select a set of hosts and make targets a view of the rest of the stuff in that list of satellites
    hosts = sats[0:4]
    targets = sats[4:]

    # Get times
    ts = load.timescale()
    tstart = ts.now()
    tend = tstart + timedelta(minutes=5) # Our time window is 3 hours long to start
    # times = ts.utc(t0.utc_datetime() + np.asarray([timedelta(minutes=x) for x in range(0, 361)])) # 360 minute (6 hour) timeframe

    observers = [Observer(h, i, 5.5/2) for i,h in enumerate(hosts)]

    opt = greedy_obs_plan_gen(observers, tstart, tend)

    print(opt)

    from datetime import datetime
    for o in opt:
        print(o.as_dict())
        o.save(f"results/{o.host_id}_{datetime.now().isoformat('T','seconds')}.json")

    # Try plotting? Need to modify the animate heatmaps dude from density such that it can take an equal length array of observation (RA,DEC) values...