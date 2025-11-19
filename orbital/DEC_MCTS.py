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

if __name__=="__main__":
    sats = load_satellites()

    import time
    start = time.perf_counter()

    # Select a set of hosts and make targets a view of the rest of the stuff in that list of satellites
    hosts = sats[0:4]
    targets = sats[4:] # Technically this is incorrect, as each telescope should look at the other hosts too!!!

