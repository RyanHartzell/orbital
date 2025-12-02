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
from functools import reduce
from kldiv_maximizer import maximize_kldiv

warnings.filterwarnings("ignore", category=RuntimeWarning)

np.random.seed(42) # We'll want to comment out once done debugging

# Constants
WORST_CASE_SLEW_PER_ACTION = np.pi
MAX_DEPTH = 100 # SMW : Adjust plannign horizion 

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
            return np.inf # High number since we want to ensure unvisited nodes are prioritized to be visited 

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
        self.children[action] = node
        return node 
    
    def check_terminal(self):
        if self.depth >= MAX_DEPTH:
            self.is_terminal = True
            return True
        return False
    
    def __repr__(self):
        return f"MCTSNode[state={self.state}, children={len(self.children)}]"
    
# Utils
def compute_access(o, t, targets):
    # t = o.last_observation_end_time
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

# find global maximum value indices
def global_argmax(arr, thresh=None):
    if thresh is None:
        # True greediness
        return np.where(np.isclose(arr, arr.max()))
    
    else:
        # Stochastic behavior
        return np.where(arr > (thresh * arr.max()))

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

        # 0.1 km is the default reset value for uncertainty (in reality this would be a kalman style update, using observation jacobian, or approximately an "imaging-pixel-error-at-range"-limited reduction in projected target covariance)
        reward += update_uncertainty(tr["last_uncertainty"], dt) - 0.1 + 10*(dt > max_allowable_unseen)

    return reward

# Cost (haversine_costs(a, b))
def compute_cost(current_state_index, new_state_index):
    return haversine_distances(np.c_[RA.flat[current_state_index],DEC.flat[current_state_index]], np.c_[RA.flat[new_state_index],DEC.flat[new_state_index]])

# Simple data classes for tracking slew actions and resulting observations (states)
class ActionStatePair:
    def __init__(self, action, state, time):
        self.action = action
        self.state = state
        self.start_time = time # Should be end_time of parent node

        # Compute end_time for ease of access
        self.end_time = time + self.action.duration + self.state.duration

class Observation:
    def __init__(self, target, integration_and_readout, exposures):
        self.duration = timedelta(seconds=integration_and_readout * exposures)
        self.target = target # This should store the target index (or flat RA/DEC index for density/belief maps)

    @classmethod
    def random_sample(cls, target):
        integration_and_readout = np.random.uniform(0.01, 5) #100fps is floor
        exposures = np.ceil(np.random.uniform(3, 17))
        return cls(target, integration_and_readout, exposures)
    
class Slew:
    def __init__(self, start_idx, end_idx, slew_rate=np.pi/4):
        self.duration = timedelta(seconds=compute_cost(start_idx, end_idx)[0,0] / slew_rate)

# Observer class (should technically live in its own module...)
class Observer:
    def __init__(self, host, host_ind):
        self.host = host
        self.host_ind = int(host_ind)

        # Sensor params
        self.afov = 5.5 # default in degrees (can make it different for different observers)

        # I should probably just build these from the observations in the chosen best path through our tree now (aka the plan)
        self.last_observation_end_time = None
        self.plan = [] # Contains flat indices into RA/DEC meshgrid (from density module) *OR* target_indices
        self.obs_starts = []
        self.obs_ends = []
        self.reward = [] # Size of plan-1, should be all associated rewards for actions
        self.cost = [] # Size of plan-1, should be all associated costs for actions

        # Leaves this up to the user for which heatmap to append
        self.maps = []
        self.map_dict = {} # should've been a dict from the start where each entry is a different type of map

        # MCTS Specific stuff
        self.sample_times = None
        self.planning_window_start = None
        self.planning_window_end = None

        # Beliefs: either probs over all target inds, or probs over all RA/DEC inds, and possibly binned over time
        # Local belief is used to bias action selection and is compared to extern belief in order to maximize relative entropy (divergence, or aka minimize mutual info)
        self.local_belief = None # observational probability over all targets at each time
        self.extern_belief = None # product distribution of each other agent's "local belief", normalized
        self.local_belief_map = None # aggregated by time ranges, projected into corresponding density maps via query results
        self.extern_belief_map = None

        # Target access masks must be stored, as do density query records and maps
        self.access = None
        self.density = None
        self.density_maps = None

        self.root = MCTSNode()
        # self.curr_node = None

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
    # Greedy helpers (rollout/simulation, init)
    def greedy_init(self, t, targets):
        # Compute access
        access = compute_access(self, t, targets)

        # # Calculate apparent ra, dec, ranges relative to host state at each time t
        obs = reformat_radecrange(calculate_apparent_radecrange(self.host, np.asarray(targets), [t], access), ragged=True)

        # Build all ball trees
        bt = construct_ball_tree(obs[0][0], obs[1][0])

        # Calculate new value map using explicit radius_query method (not KDE, since we want total value based on target indices)
        density_map, query_results = construct_fov_density_map(bt)

        # Pick best DENSITY!!! index (argmax)
        new_state_index = np.ravel_multi_index(global_argmax(density_map), density_map.shape)[0] # I can change this to a random sample instead...

        self.root = MCTSNode(
            state=ActionStatePair(
                action=Slew(new_state_index, new_state_index),
                state=Observation(
                    target=new_state_index, # Flattened RA/DEC targeting index
                    integration_and_readout=0.0,
                    exposures=1
                ),
                time=t
            ),
            parent=None,
        )


    ######################################################################
    # Beginning of Dec-MCTS implementation

    # Select an action given state
    def choose_action(self, node):
        # Here we'll sample from local belief map given node's observation end_time as our new start

        # find nearest local belief map
        nn_ind = self.find_nearest(node.state.end_time, self.sample_times)

        # RH: EPSILON-GREEDY ON BELIEF FOR NOW!!!
        # get new_state_index via epsilon-greedy global argmax sampling, or just straight up random sample from belief map?
        action_inds = np.ravel_multi_index(global_argmax(self.local_belief_map[nn_ind], 0.8), self.local_belief_map[nn_ind].shape)

        # Choose a new index from the best inds
        new_state_index = np.random.choice(action_inds) # Or randomly select np.random.choice(action_inds)

        # Form ActionStatePair to transition from old state to new state
        asp = ActionStatePair(
            action=Slew(
                start_idx=node.state.state.target,
                end_idx=new_state_index,
            ),
            state=Observation.random_sample(
                target=new_state_index
            ),
            time=node.state.end_time
        )

        return asp

    # ALL OF THESE BELIEF FUNCTIONS COMPUTE BELIEF AT TIME STEPS PROVIDED BY GLOBAL PLANNER!!!
    def compute_local_belief(self, targets):
        # RH: INIT LOCAL BELIEF TO UNIFORM DISTRIBUTION!!!!!
        # Use tree to get observational probabilities across all targets using top-k highest value paths through tree
        # Given top-k highest value paths, bin by OBSERVATION END TIMES

        # RH: For now, just use the very best action sequence
        path = self.get_best_action_sequence(mode="duct") # we'd like this to return topk sequences instead...

        # Using subsets of observations for each time range, compute probabilities over targets by the frequency at which they show up (given state and corresponding self.density query record)
        self.local_belief = np.zeros((len(self.sample_times), len(targets)))
        for node in path:
            nn = self.find_nearest(node.state.end_time, self.sample_times)
            qr = self.density[nn][node.state.target]
            self.local_belief[nn][self.access[nn][:,0]][qr] += 1 # I think here we'd also divide by the number of paths we're drawing from aka "k"
        
        for lb in self.local_belief:
            # Check for rare case of zeros (might happen with non-terminal paths or finely sampled times)
            if np.isclose(s:=lb.sum(),0.0):
                lb[:] = 1.0/lb.size
            lb /= s # Normalize

    def compute_belief_maps(self):
        # Using targeting belief and density lookup, aggregate into spatial map at each time t, which will be used for action selection
        self.local_belief_map = [np.zeros_like(self.density_maps[i])]*len(self.sample_times)
        self.extern_belief_map = [np.zeros_like(self.density_maps[i])]*len(self.sample_times)

        for i in range(len(self.local_belief)):
            # For all queries across all belief times
            for j,qr in enumerate(self.density[i]):
                self.local_belief_map[i].flat[j] += np.sum(self.local_belief[i][self.access[i][:,0]][qr])
                self.extern_belief_map[i].flat[j] += np.sum(self.extern_belief[i][self.access[i][:,0]][qr])
            # Normalize!
            self.local_belief_map[i] /= self.local_belief_map[i].sum()
            self.extern_belief_map[i] /= self.extern_belief_map[i].sum()

    def optimize_belief(self, extern):
        extern = np.asarray(extern)
        # extern.shape = [# observers, # times, # targets]
        # For each time t
        for i in range(len(self.sample_times)):
            # Aggregate external beliefs (top-k targeting probability vectors)
            # Combine extern belief via product distribution and normalization
            extern_belief = reduce(np.multiply, extern[:,i,:])
            extern_belief[extern_belief < 0] = 0.0 # clip
            extern_belief = extern_belief / extern_belief.sum() # normalize
            self.extern_belief[i] = extern_belief

            # Maximize KL-Divergence of P1=local vs P2=extern
            opt = maximize_kldiv(self.local_belief[i], extern_belief)

            # Append to local belief vector (or index into and replace belief if times are fixed for planning)
            self.local_belief[i] = opt

    def compute_density(self, targets):
        # This should compute all density map query records for times
        self.access = [None]*len(self.sample_times)
        self.density = [None]*len(self.sample_times)
        self.density_maps = [None]*len(self.sample_times)

        for i,t in enumerate(self.sample_times):
            # Compute access
            self.access[i] = compute_access(self, t, targets)

            # # Calculate apparent ra, dec, ranges relative to host state at each time t
            obs = reformat_radecrange(calculate_apparent_radecrange(self.host, np.asarray(targets), [t], self.access[i]), ragged=True)

            # Build all ball trees
            bt = construct_ball_tree(obs[0][0], obs[1][0])

            # Calculate new value map using explicit radius_query method (not KDE, since we want total value based on target indices)
            self.density_maps[i], self.density[i] = construct_fov_density_map(bt, self.afov)
    
    @staticmethod
    def find_nearest(t, times):
        return np.searchsorted(times, t)

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

        node.check_terminal() # Check depth first
        self.mcts_check_terminal(node) # Given state associated with node, determine if end_time for the observation at that state is after planning horizon end

        if node.is_terminal:
            return None # Reach terminal state 
        
        for a in range(num_children): # Static node width so just make num_children amount of state action pairs 
            #next_state = np.random.randint(0,100) # SMW:  Replace this with actual state 
            next_state = self.choose_action(node)
            node.add_child(a,next_state)

        # From the nodes we created randomly select one to work from
        return random.choice(list(node.children.values()))

    # TODO: RH - WE NEED TO FIGURE OUT HOW TO EFFICIENTLY SIMULATE ROLLOUT!!!! Could look like a full greedy selection of actions, or random selection given local belief?    
    def simulate(self,node):
        # Dummy rollout: this needs to pick random (or greedy?) actions across the remaining density maps as an approximation?

        if node.is_terminal:
            return 0.0
        
        # TODO: This must be updated to simulate via epsilon-greedy, random belief-driven, or random uncertainty-biased belief-driven action sampling "policies"
        return np.random.uniform(0, 1)
    
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
        if not leaf.is_terminal and not leaf.children:
            tmp = self.expand(leaf)
            if tmp:
                leaf = tmp
                path.append(leaf)
        
        # 3. Simulate
        reward = self.simulate(leaf)

        # 4. backpropagate 
        self.backpropagate(path, reward)

        # return reward
    
    def mcts_check_terminal(self, node):
        # We need to check terminal nodes by end_time vs end of planning horizon
        # Set node.is_terminal = True if end_time of assigned observation is after end of planning window
        if node.is_terminal:
            return

        # Use end_time on ActionStatePair object, since that has end_time calculated via: end_time = parent.end_time + slew duration + observation duration = ActionStatePair.end_time
        if node.state.end_time > self.planning_window_end:
            node.is_terminal = True
    
    def get_best_action_sequence(self, mode="duct", k=1):
        path = []

        # RH: Is this one literally just given by the D-UCT method?
        if mode=="prob":
            # Choose highest probability sequence: argmax(q_i) from paper
            path = []

        if mode=="duct":
            leaf, path = self.select_node()

        if mode=="disc_value":
            node = self.root
            while node.children:
                new_node = max(node.children.values(), key=lambda x: x.disc_value)
                path.append(new_node)
                node = new_node
            # score = sum([n.disc_value for n in path])

        return path

    # RH: Just keeping this comment block for notes to myself
    # def compute_joint_action_beliefs(self, beliefs, times):
    #     # Using incoming beliefs from other observers (action probs, action chain) update own belief

    #     # Three ways to do this:
    #     #   a) Bias target selection by updating all target uncertainties with BEST OBS SEQUENCE FROM EACH OTHER HOST
    #     #   b) Bias target selection by choosing randomly from INVERSE OF COMBINED TOP-K ACTION SEQUENCE OBSERVATION PROBABLITY VECTOR FROM ALL OTHER HOSTS
    #     #           - May require binning by time step to match density maps
    #     #           - Average would also be valid I think, as long as I renormalize probs
    #     #   c) Bias RA/DEC selection by choosing randomly from SPATIALLY AGGREGATED COMBINED TOP-K ACTION SEQUENCE OBSERVATION PROBABLITY VECTOR FROM ALL OTHER HOSTS
    #     #           - Requires binning by time step to match density maps
    #     #
    #     return

    def reset(self):
        # Reset tree state to last known observation for observer, or sub-tree under a specific node
        return

# Global Planner (Simulates the comms between local planners on agents!!!!)
class GlobalDecMCTSPlanner:
    def __init__(self, plan_start, dt=timedelta(minutes=5), DT=timedelta(minutes=30), plan_horizon=timedelta(hours=1.5), plan_duration=timedelta(hours=3)):
        # Time deltas
        self.dt = dt # This controls how many density maps or bins we have over time for beliefs
        self.DT = DT # Plan window shift
        self.plan_horizon = plan_horizon
        self.plan_duration = plan_duration
        self.plan_start = plan_start

        # Planning window end will update over time with horizon and DT shift
        self.planning_window_start = plan_start
        self.planning_window_end = plan_start + plan_horizon

        # Planning results stores {observer: [path.copy(), ...]}
        self.results = {}

        # Working variables
        self.observers = None
        self.targets = None

    # Init all planners greedily
    def setup(self, observers, targets):
        self.observers = observers.copy()
        self.targets = targets

        # Calculate density at start time for each observer
        for o in self.observers:
            times = [self.planning_window_start + i * self.dt for i in range(self.plan_horizon // self.dt + 1)]
            o.sample_times = times
            o.planning_window_start = self.planning_window_start
            o.planning_window_end = self.planning_window_end

            # All of this could probably be triggered from a setup function on the observer?
            # Check syntax
            o.compute_density(self.targets)

            # Set beliefs and belief maps to uniform distributions?
            o.init_belief(self.targets)

            # Choose initial pointing greedily
            o.greedy_init(self.targets)

        # Set up target records?
        # Init target records
        # TODO: RH - these will likely need to be computed on-the-fly when doing rollout or choosing actions :/
        # self.target_records = np.asarray([{"last_seen": self.planning_window_start, "last_uncertainty": 1.0} for _ in targets])

    # This should include belief update for our observers/local planners    
    def run(self, nsync=10, niter=1000):
        # For number of communication rounds (aka 5 to allow convergence?), do chunk of mcts iterations
        for _ in range(nsync):
            # For each observer, start thread to run mcts search function (aka mcts_iter in loop)
            with threadpool as tp: # pseudo code
                # Do mcts_iter
                for i in range(niter):
                    #


            # Join threads, accumulate beliefs, update beliefs on each observer

            # Reset trees?


        # RH: FOR NOW WE'RE JUST RUNNING A SINGLE TIME HORIZON WHERE EACH STEP IS MCTS_ITER FOLLOWED BY BELIEF COMMS AND UPDATE!!!!!

        # BEFORE RETURNING, UPDATE ALL TIMING VARIABLES!!! Namely planning_window_end, which is used to check termination of MCTS paths
        # RH: Basically add DT to planning window start and end
        # self.planning_window_start += self.DT
        # self.planning_window_end += self.DT

        return
    
    # Trigger reset across all local planners    
    def reset(self):
        for o in self.observers:
            o.reset()

if __name__=="__main__":
    obs = Observer(host = None, host_ind=0)

    for i in range(1000):
        r = obs.mcts_iter()

    print(obs.get_best_action_sequence())

    # Change this to load from a text file on disk instead of download, and set start time to the time in metadata.txt
    # sats = load_satellites()

    # import time
    # start_init = time.perf_counter()

    # # Select a set of hosts and make targets a view of the rest of the stuff in that list of satellites
    # hosts = sats[0:4]
    # targets = sats[4:100] # Technically this is incorrect, as each telescope should look at the other hosts too!!!


    # # Init time:
    # end_init = time.perf_counter() - start_init