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
import random



warnings.filterwarnings("ignore", category=RuntimeWarning)

# Constants
WORST_CASE_SLEW_PER_ACTION = np.pi
MAX_DEPTH = 20 # SMW : Adjust plannign horizion 


class MCTSNode:
    def __init__(self, state = None, parent=None, gamma=0.99, depth = 0):
        self.state = state
        self.parent = parent
        self.children = {}

        # Discount factor 
        self.gamma = gamma

        #Exploration Constant
        self.c = 0.1

        # Discounted statistics
        self.disc_visits = 0.0        # C_t(s)
        self.disc_value = 0.0         # discounted return estimate

        # For standard MCTS if needed
        self.visits = 0
        self.value = 0.0

        # Depth and terminal state if we force a terminal statet
        self.depth = depth 
        self.is_terminal = False
    
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
            return 1e4 # High number since we want to ensure unvisited nodes are prioritized to be visited 

        exploitation = self.disc_value / self.disc_visits
        parent_visits_safe = max(parent.disc_visits, 1.0) # Clamp input for log to be at least 1
        exploration = self.c * math.sqrt(
            math.log(parent_visits_safe) / self.disc_visits
        )

        return exploitation + exploration
    def best_child(self):
        # Since we handle the children for each node we can iterate from this function and find the max
        if not self.children:
            return None
        return max(self.children.values(), key = lambda child: child.d_uct(self))
    
    def add_child(self, action, next_state):
        node = MCTSNode(
            state=next_state, 
            parent = self, 
            gamma = self.gamma,
            depth=self.depth + 1
        )
        self.children[action] = node;
        return node 
    
    def check_terminal(self):
        if self.depth >= MAX_DEPTH:
            self.is_terminal = True
            return True
        return False
    
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

    # SMW added functions for selecting/expanding tree
    def select_node(self):
        # So traverse the tree using DUCT until we reach a leaf node
        node = self.root
        path = [node]

        while node.children and not node.is_terminal:
            child = node.best_child()
            if child is None:
                break
            node = child
            path.append(node)
        
        return node, path 
    
    def expand(self, node, num_children = 5):
        # Expand by creating random next states (right now I set to 5 default )

        node.check_terminal()

        if node.is_terminal:
            return None # Reach terminal state 
        
        for a in range(num_children): # Static node width so just make num_children amount of state action pairs 
            next_state = np.random.randint(0,100) # SMW:  Replace this with actual state 
            node.add_child(a,next_state).depth= node.depth + 1
        # From the nodes we created randomly select one to work from 
        return random.choice(list(node.children.values()))
    
    def simulate(self,node):
        # Dummy rollout

        if node.is_terminal:
            return 0.0
        
        return np.random.uniform(-1, 1)
    
    def backpropagate(self, path, reward):
        # Discounted backip each step we apply a discount
        for node in reversed(path):
            node.update_visit(reward)
            if node.parent is None:
                continue
            node.disc_visits *= node.gamma
            node.disc_value  *= node.gamma
    def mcts_iter(self):
        # 1. Selection
        leaf,path = self.select_node()

        # 2. Expansion
        if not leaf.children:
            leaf = self.expand(leaf)
            path.append(leaf)
        
        # 3. Simulate
        reward = self.simulate(leaf)

        # 4. backpropagate 
        self.backpropagate(path, reward)

        return reward
    
    
if __name__=="__main__":
    obs = Observer(host = None, host_ind=0)

    for i in range(1000):
        r = obs.mcts_iter()
        print(f"Iter {i}: reward={r:.3f}")

    #sats = load_satellites()

    #import time
    #start = time.perf_counter()

    # Select a set of hosts and make targets a view of the rest of the stuff in that list of satellites
    #hosts = sats[0:4]
    #targets = sats[4:] # Technically this is incorrect, as each telescope should look at the other hosts too!!!

