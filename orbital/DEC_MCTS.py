"""
This uses all our wonderful access and density calculations to create a dec-mcts 
"""
from density import *
from access import in_major_keep_out_zones, not_sunlit, out_of_range
from sklearn.metrics.pairwise import haversine_distances
import numpy as np
# from skimage.filters import peak_local_max 
from datetime import timedelta
from astropy import units as u
from skyfield.api import load
import json
import warnings
import math

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Constants
WORST_CASE_SLEW_PER_ACTION = np.pi

# Utils
def compute_access(o, targets):
    t = o.last_observation_end_time
    host = o.host

    # Get access mask (THIS WE SHOULD ACCELERATE AND PRECOMPUTE!!!)
    sunlit_access = not_sunlit(t, targets)
    # print(f"% access [SUNLIT] = {np.sum(~sunlit_access)/sunlit_access.size * 100.}")

    range_access = out_of_range(t, host, targets)
    # print(f"% access [IN-RANGE] = {np.sum(~range_access)/range_access.size * 100.}")

    koz_access = in_major_keep_out_zones(t, host, targets)
    # print(f"% access [NOT-IN-KOZ] = {np.sum(~koz_access)/koz_access.size * 100.}")

    # Construct overall access mask (should be SATNUM x TIMESTEP)
    access = ~sunlit_access * ~range_access * ~koz_access # We can multiply these since any zero value should cause a switch to False
    # print(f"Total % access across timesteps = {np.sum(access)/access.size * 100.}")

    # Boolean mask for targets, True: Accessible, False: Inaccessible
    return access

# Uncertainty update (U0 is just km, dt is expected to be a timedelta object)
def update_uncertainty(U0, dt, rate=0.1/3600): # rate is 0.1 km/h converted to km/s 
    return U0 + rate*U0*dt.total_seconds()

# Integration time + slew time
# def obs_duration(slew, avg_target_distance):
def obs_duration(slew, slew_rate=np.pi/4, frames=7, integration=1):
    # For simplicity (slew in radians / slew rate in rad/s) + (frames unitless * integration in s) = duration in s
    return slew / slew_rate + frames * integration

# Value (observation/collection quality)
# Target records should just be a list of the same length as the full targets list, but with a dictionary of data stored at each element (essentially per target)
def compute_reward(t, target_records, access, query_result, max_allowable_unseen=timedelta(hours=0.5)):
    # Get subset of target record dictionaries
    targ_recs = target_records[access[:,0]][query_result] # Query result is flattened index w.r.t accessible targets

    # Compute the absolute magnitude uncertainty reduction for all targets
    # Compute the absolute magnitude change in staleness index for all targets
    reward = 0.0
    for tr in targ_recs:
        dt = timedelta(seconds=(t - tr["last_seen"])) # Timedelta, which I believe is evaluated as seconds

        # 0.1 km is the default reset value for uncertainty
        # TODO: Change this to our value function for maximizing reductions in uncertainty
        reward += (update_uncertainty(tr["last_uncertainty"], dt) - 0.1 + 10*(dt > max_allowable_unseen))

    return reward

# Cost (haversine_costs(a, b))
def compute_cost(current_state_index, new_state_index):
    return haversine_distances(np.c_[RA.flat[current_state_index],DEC.flat[current_state_index]], np.c_[RA.flat[new_state_index],DEC.flat[new_state_index]])

# Observer class (should technically live in its own module...)
class Observer:
    def __init__(self, host, host_ind):
        self.host = host
        self.host_ind = int(host_ind)
        self.last_observation_end_time = None
        self.plan = [] # Contains flat indices into RA/DEC meshgrid (from density module)
        self.obs_starts = []
        self.obs_ends = []
        self.reward = [] # Size of plan-1, should be all associated rewards for actions
        self.cost = [] # Size of plan-1, should be all associated costs for actions

        # Leaves this up to the user for which heatmap to append
        self.maps = []

        self.root = MCTSNode()
        self.curr_node = None

    def __lt__(self, other):
        return self.last_observation_end_time < other.last_observation_end_time

    def as_dict(self):
        return {
            "Index": self.host_ind,
            "Name": self.host.name,
            "Plan": self.plan,
            "StartTimes": self.obs_starts, 
            "EndTimes": self.obs_ends,
            "Rewards": self.reward,
            "TotalReward": float(np.sum(self.reward)),
            "Costs": self.cost,
            "TotalCost": float(np.sum(self.cost))
        }

    def save(self, fname):
        # This should save to disk whatever Observer data we want as a pandas dataframe CSV maybe?
        with open(fname, 'w') as f:
            f.write(json.dumps(self.as_dict())) # TAKE CARE CONVERTING NUMPY TYPES TO JSON!!! Must be raw python types for base serializer to work

    def save_maps(self, fname):
        np.savez(fname, np.dstack(self.maps))

######################################################################
# Beginning of Dec-MCTS implementation
class MCTSNode:
    def __init__(self, state, parent=None, gamma=0.99):
        self.state = state
        self.parent = parent
        self.children = {}

        # Discount factor 
        self.gamma = gamma

        #Exploration Constant
        self.c = 1

        # Discounted statistics
        self.disc_visits = 0.0        # C_t(s)
        self.disc_value = 0.0         # discounted return estimate

        # For standard MCTS if needed
        self.visits = 0
        self.value = 0.0
    
    def decay(self): # this should be called every global timestep even if the node is not visited
        # Apply a decay when the node is not visited 
        self.disc_visits *= self.gamma # C_t(s)
        self.disc_value *= self.gamma # sum
    
    def update_visit(self, reward):
        # Apply the discounted update when we visit the node
        self.disc_visits = self.gamma * self.disc_visits + 1
        self.disc_value = self.gamma * self.disc_value + reward
        self.visits += 1
        self.value += reward
    
    def d_uct(self, parent):
        # Safety: avoid divide by 0 early in search
        if parent.disc_visits == 0 or self.disc_visits == 0:
            return float("inf")

        exploitation = self.disc_value / parent.disc_visits
        exploration = self.c * math.sqrt(
            math.log(self.disc_visits) / parent.disc_visits
        )

        return exploitation + exploration

# Local Planner (Runs on each agent concurrently, hopefully multi-threaded)
class LocalDecMCTSPlanner:
    def __init__(self, observer, targets, update_period, schedule_duration, p_comms_dropout=0.5):
        self.o = observer
        self.targs = targets

    def setup(self):
        # Init tree greedily for first observation, and then build from that node
        return

    def update_belief(self):
        # Using incoming beliefs from other observers (action probs, action chain) update own belief

        # Avg of incoming beliefs, or should update as latest observed for each target out of set of beliefs?

        return

    # Triggered every update period for a fresh look at densities/values
    def compute_density(self, times):
        # This should compute density over time and store time-derivatives for interp
        for t in times:
            # Compute apparent RA/DECs
            # Compute ball-tree
            # Compute density map

        return
    
    def interpolate_density(self):
        # given a time, lookup nearest neighbors and then do bilinear interp on sets to get value/cost
        return
    
    def compute_action_sequence_probs(self):
        # We need to update these as we explore the tree with best leaf actions, right?
        return
    
    def step(self):
        # Action selection
        return
    
    def reset(self):
        # Reset tree state to last known observation for observer
        return

# Global Planner (Simulates the comms between local planners on agents!!!!)
class GlobalDecMCTSPlanner:
    def __init__(self):
        pass

    # Init all planners greedily
    def setup(self):
        return

    # This should include belief update for our observers/local planners    
    def step(self):
        return

    # Trigger reset across all local planners    
    def reset(self):
        return

if __name__=="__main__":
    sats = load_satellites()

    import time
    start_init = time.perf_counter()

    # Select a set of hosts and make targets a view of the rest of the stuff in that list of satellites
    hosts = sats[0:4]
    targets = sats[4:] # Technically this is incorrect, as each telescope should look at the other hosts too!!!


    # Init time:
    end_init = time.perf_counter() - start_init