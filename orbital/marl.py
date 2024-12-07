"""
This uses all our wonderful access and density calculations to create a RL'd plan
"""
from density import *
from access import in_major_keep_out_zones, not_sunlit, out_of_range
from sklearn.metrics.pairwise import haversine_distances
import numpy as np
# from skimage.filters import peak_local_max 
from datetime import timedelta
from astropy import units as u

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import torch.nn.functional as F

import warnings
from tqdm import tqdm
import json
from copy import deepcopy

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

        # Flattened RL observation vectors of concatenated environment density map, current position index (density_grid.flat + max_uncertainty_grid.flat + plan[-1])
        self.rl_observation_vectors = []

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
    
    def clear(self):
        self.last_observation_end_time = None
        self.plan = []
        self.obs_starts = []
        self.obs_ends = []
        self.reward = []
        self.cost = []
        self.maps = []
        self.rl_observation_vectors = []

    def save(self, fname):
        # This should save to disk whatever Observer data we want as a pandas dataframe CSV maybe?
        with open(fname, 'w') as f:
            f.write(json.dumps(self.as_dict())) # TAKE CARE CONVERTING NUMPY TYPES TO JSON!!! Must be raw python types for base serializer to work

    def save_maps(self, fname):
        np.savez(fname, np.dstack(self.maps))

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
def global_argmax(arr, thresh=None):
    if thresh is None:
        # True greediness
        return np.where(np.isclose(arr, arr.max()))
    
    else:
        # Stochastic behavior
        return np.where(arr > (thresh * arr.max()))

# GREEDY OBS PLANNER
# def greedy_obs_plan_gen(observers, time_window_start, time_window_end):
#     # For each action, FOR EACH SENSOR, choose the highest value, lowest cost observation to make. Uses global armax with associated values and costs and then lexicographically sorts by value then by cost
#     observers = observers.copy()
#     final = []

#     # Get initial states for each observer (just straight up best move as of time_window_start)
#     for o in observers:
#         o.last_observation_end_time = time_window_start
#         init_greedy(o, targets)

#     # Init target records
#     target_records = np.asarray([{"last_seen": time_window_start, "last_uncertainty": 1.0} for _ in targets])

#     while observers:
#         observers = sorted(observers) # __lt__ should be used to compare two observers by latest_timestamp (end of previous observation), with earliest going first
#         for i,o in enumerate(observers):
#             # do execute greedy step

#             # Observer data should be updated at the end of this (reward, cost, new state, last observation end time)
#             # TargetRecords should be updated at the end of this function as well (last seen time, last uncertainty)
#             execute_greedy_step(o, targets, target_records)

#             if o.last_observation_end_time > time_window_end:     
#                 final.append(observers.pop(i))

#     # All observation chains are stored on the objects returned in final (which may be in a differnt order than the initial observer list)
#     return final

# Uncertainty update (U0 is just km, dt is expected to be a timedelta object)
def update_uncertainty(U0, dt, rate=0.1/3600): # rate is 0.1 km/h converted to km/s 
    return U0 + rate*U0*dt.total_seconds()

# Integration time + slew time
# def obs_duration(slew, avg_target_distance):
def obs_duration(slew, slew_rate=np.pi/4, frames=7, integration=5):
    # For simplicity (slew in radians / slew rate in rad/s) + (frames unitless * integration in s) = duration in s
    return slew / slew_rate + frames * integration

# Value (observation/collection quality)
# Target records should just be a list of the same length as the full targets list, but with a dictionary of data stored at each element (essentially per target)
def compute_reward(t, target_records, access, query_result, max_allowable_unseen=timedelta(hours=0.5)):
    # Get subset of target record dictionaries
    targ_recs = target_records[access[:,0]][query_result] # Query result is flattened index w.r.t accessible targets

    # Compute dt = t - target_records.last_seen
    # Compute the absolute magnitude uncertainty reduction for all targets
    # Compute the absolute magnitude change in staleness index for all targets
    reward = 0.0
    for tr in targ_recs:
        dt = timedelta(seconds=(t - tr["last_seen"])) # Timedelta, which I believe is evaluated as seconds

        # 0.1 km is the default reset value for uncertainty
        reward += 10 * (update_uncertainty(tr["last_uncertainty"], dt) - 0.1 + (dt > max_allowable_unseen))

    return reward

# Cost (haversine_costs(a, b))
def compute_cost(current_state_index, new_state_index):
    return haversine_distances(np.c_[RA.flat[current_state_index],DEC.flat[current_state_index]], np.c_[RA.flat[new_state_index],DEC.flat[new_state_index]])

def init_greedy(o, targets):
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

# My version of Gridworld problem adapted to distributed telescope problem
from collections import defaultdict

# class TelescopeMDP(MDP):
class TelescopeMDP:
    def __init__(
        self,
        observers,
        targets,
        start_time,
        end_time,
        state_space,
        action_space,
        discount_factor=0.9,
    ):
        self.observers = observers
        self.targets = targets
        self.discount_factor = discount_factor

        # indices of our state grid
        self.state_space = state_space
        self.action_space = action_space
        self.rlobsvec_prealloc = np.zeros(self.state_space)

        # This will store all resulting agent records (plans, rewards, costs, etc. per step in a given episode)
        self.episode_metadata = []

        # HERE WE NEED TO DO THE REST OF THE SETUP/INITIALIZATION DONE AS PART OF THE PLAN GEN FUNCTION
        self.time_window_start = start_time
        self.time_window_end = end_time

        self.target_records = None
        self._reset()

    # Initial setup (run as part of episode reset!)
    def _reset(self):
        # Get initial states for each observer (just straight up best move as of time_window_start)
        for o in self.observers:
            o.clear() # empty all records
            o.last_observation_end_time = self.time_window_start
            init_greedy(o, self.targets) # initial state stored on each observer object!!! NOT ON THE ENV!!!!!!

        # Init target records
        self.target_records = np.asarray([{"last_seen": self.time_window_start, "last_uncertainty": 1.0} for _ in self.targets])

        # Clear all other environment state and records
        self.rlobsvec_prealloc = np.zeros(self.state_space)

    # Clears training record as well
    def _hard_reset(self):
        self._reset()
        self.episode_metadata = []

    # We should overload the execute_policy function to just run this once per episode
    def run_episode(self, policies):
        self._reset()
        observers = self.observers.copy()
        final = []
        step = 0

        # Can we get tqdm working here???? Problem for another day tbh
        while observers:
            observers = sorted(observers) # __lt__ should be used to compare two observers by latest_timestamp (end of previous observation), with earliest going first
            for i,o in enumerate(observers):
                # Observer data should be updated at the end of this (reward, cost, new state, last observation end time)
                # TargetRecords should be updated at the end of this function as well (last seen time, last uncertainty)
                self.execute(o, policies[o.host_ind], step) # This will be analogous to the execute_greedy_step function

                if o.last_observation_end_time > self.time_window_end:     
                    final.append(observers.pop(i))
            step += 1

        # Here's where we record all episode information (observers have reward, cost, and cumulative information)
        self.episode_metadata.append(final)

    # def get_reward(self, state, action):
    #     reward = compute_reward() - compute_cost(state, action)
    #     step = len(self.episode_rewards)
    #     self.episode_rewards += [reward * (self.discount_factor ** step)]
    #     return reward

    def get_discount_factor(self):
        return self.discount_factor

    # SINGLE STEP EXECUTION WITH ACTIVE POLICY!
    def execute(self, observer, policy, step):
        t = observer.last_observation_end_time
        host = observer.host

        # print(f"[INFO] Now processing {o.host} @ time = {t}")

        # Get access mask
        sunlit_access = not_sunlit(t, self.targets)
        range_access = out_of_range(t, host, self.targets)
        koz_access = in_major_keep_out_zones(t, host, targets)

        # Construct overall access mask (should be SATNUM x TIMESTEP)
        access = ~sunlit_access * ~range_access * ~koz_access # We can multiply these since any zero value should cause a switch to False

        # Calculate apparent ra, dec, ranges relative to host state at each time t
        obs = reformat_radecrange(calculate_apparent_radecrange(host, np.asarray(targets), [t], access), ragged=True)

        # Build all ball trees
        bt = construct_ball_tree(obs[0][0], obs[1][0])

        # Calculate new value map using explicit radius_query method (not KDE, since we want total value based on target indices)
        # density_map, query_results = construct_fov_density_map(bt)
        density_map, query_results = construct_fov_density_map(bt) # This will generate a dstack'd spatial feature map which is shaped like (*spatial_grid_shape, n_features) and initially represents just uncertainty and density(occupancy)

        # Compute max uncertainty of the target records at each grid point
        uncertainty_map = np.zeros(density_map.size)
        for qi, q in enumerate(query_results):
            # get max uncertainty for each grid point
            if (len(q) > 0):
                # Downselect the target records and map across them to get the max value of the uncertainties at the current time
                uncertainty_map[qi] = max(map(lambda x: update_uncertainty(x['last_uncertainty'], timedelta(seconds=(t - x["last_seen"]))), self.target_records[access[:,0]][q]))

        if np.any(access):
            # First we construct our rl_observation_vector
            self.rlobsvec_prealloc[...] = np.concatenate([density_map.flat, uncertainty_map, [observer.current_state_index]])

            # Instead of exhaustively computing value map to optimize on as a heuristic, we choose a new state by sampling the policy
            new_state_index = policy.select_action(self.rlobsvec_prealloc)

            # Technically should be moving this up into the value_map builder above, but assuming slew is negligible for now...
            slew = compute_cost(observer.current_state_index, new_state_index)
            duration = timedelta(seconds=obs_duration(slew)[0,0])

        else:
            new_state_index = observer.current_state_index
            slew = 0.0
            duration = timedelta(seconds=0.0)

        if len(slew) != 0:
            slew = slew.flat[0]
    
        # Now update all the stuff we need on the observer!
        observer.current_state_index = new_state_index
        observer.last_observation_end_time = t + duration

        # Update observer traces
        observer.plan.append(int(new_state_index))
        observer.obs_starts.append(t.utc_iso())
        observer.obs_ends.append(observer.last_observation_end_time.utc_iso())

        # Reformulating so that we don't explicitly optimize on rewards!!!!! So we compute the reward of the chosen action of the policy
        # WE SHOULD APPLY THE DISCOUNT TO THE REWARD HERE I THINK!!!!
        observer.reward.append(float((compute_reward(t, self.target_records, access, query_results[new_state_index]) - float(slew/WORST_CASE_SLEW_PER_ACTION)) * (self.discount_factor ** step)))
        
        observer.cost.append(float(slew/WORST_CASE_SLEW_PER_ACTION))

        # In this case, let's dump in the density map for viz purposes
        observer.maps.append(density_map) # 0th index is density, 1st index is max_uncertainty
        observer.rl_observation_vectors.append(self.rlobsvec_prealloc)

        # For each target, update our global target record with last seen time and updated uncertainty!!!
        for tr in self.target_records[access[:,0]][query_results[new_state_index]]:
            # tr["last_uncertainty"] = update_uncertainty(tr["last_uncertainty"], timedelta(seconds=(t - tr["last_seen"])))
            tr["last_uncertainty"] = 0.1
            tr["last_seen"] = observer.last_observation_end_time


# No changes needed probably!!!
class DeepNeuralNetworkPolicy:
    """
    An implementation of a policy that uses a PyTorch (https://pytorch.org/) 
    deep neural network to represent the underlying policy.
    """

    def __init__(self, state_space, action_space, hidden_dim=64, alpha=0.001, stochastic=True):
        self.state_space = state_space
        self.action_space = action_space

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running DeepNeuralNetworkPolicy on {'GPU' if torch.cuda.is_available() else 'CPU'}")

        # Define the policy structure as a sequential neural network.
        self.policy_network = nn.Sequential(
            nn.Linear(in_features=self.state_space, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=self.action_space),
        )

        # Move network to GPU!
        self.policy_network.to(self.device)

        # Initialize weights using Xavier initialization and biases to zero
        self._initialize_weights()

        # The optimiser for the policy network, used to update policy weights
        self.optimiser = Adam(self.policy_network.parameters(), lr=alpha)

        # Whether to select an action stochastically or deterministically
        self.stochastic = stochastic

    def _initialize_weights(self):
        for layer in self.policy_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Ensure the last layer outputs logits close to zero
        last_layer = self.policy_network[-1]
        if isinstance(last_layer, nn.Linear):
            with torch.no_grad():
                last_layer.weight.fill_(0)
                last_layer.bias.fill_(0)

    """ Select an action using a forward pass through the network """

    def select_action(self, state):
        # I like the 'observation' terminology better, where this incoming state is an observation of the global environment state concatenated with our agent data (position index)
        # i.e. obs = flattened global density grid + flattened max uncertainty grid + current agent position (boresight index) = state vector 'state'

        # Convert the state into a tensor so it can be passed into the network
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action_logits = self.policy_network(state) # This should be an observation of the entire state space, possibly concatenated with our current position or affected by a cosine^4 or 1/r^2 falloff or something centered on our location??????

        action_distribution = Categorical(logits=action_logits)
        # Sample an action according to the probability distribution
        return action_distribution.sample().item() # This should give us the new position of the boresight!!!!

    """ Get the probability of an action being selected in a state """
    def get_probability(self, state, action):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_logits = self.policy_network(state)

        # A softmax layer turns action logits into relative probabilities
        probabilities = F.softmax(input=action_logits, dim=-1).tolist()
        # Convert from a tensor encoding back to the action space
        return probabilities[action]

    # This function evaluates a full episode trajectory!!!!
    def evaluate_actions(self, states, actions):
        action_logits = self.policy_network(states)
        action_distribution = Categorical(logits=action_logits)
        log_prob = action_distribution.log_prob(actions.squeeze(-1))
        return log_prob.view(1, -1)

    def update(self, states, actions, deltas):
        # Convert to tensors to use in the network
        deltas = torch.as_tensor(deltas, dtype=torch.float32, device=self.device)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)

        action_log_probs = self.evaluate_actions(states, actions)

        # Construct a loss function, using negative because we want to descend,
        # not ascend the gradient
        loss = -(action_log_probs * deltas).sum()
        self.optimiser.zero_grad()
        loss.backward()

        # Take a gradient descent step
        self.optimiser.step()

    def save(self, filename):
        torch.save(self.policy_network.state_dict(), filename)

    @classmethod
    def load(cls, state_space, action_space, filename):
        policy = cls(state_space, action_space)
        policy.policy_network.load_state_dict(torch.load(filename))
        return policy
    
# AKA: REINFORCE algorithm referenced in the notes!!!!!!!! Just renamed in the code

# This is going to run a version of our plan_gen function as execute function
class PolicyGradientExecutor:
    def __init__(self, mdp, policies) -> None:
        self.mdp = mdp
        self.policies = policies

        if len(self.mdp.observers) != len(self.policies):
            raise Exception(f"Number of observers managed by the MDP ({len(self.mdp.observers)}) is different from the number of policies supplied ({self.policies}). These must match.")

    """ Generate and store an entire episode trajectory to use to update the policy """

    def execute(self, episodes=100):

        # Clear the existing episode_metadata (training_history might be a better name) before executing a new training run
        self.mdp.episode_metadata = []
        self.reward_traces = np.zeros((len(self.mdp.observers), episodes))

        for episode in tqdm(range(episodes)):

            # This appends data to self.mdp.episode_metadata list of all episode observer records for every episode
            self.mdp.run_episode(self.policies)

            # For each observer, update its corresponding policy by ID
            for o in self.mdp.episode_metadata[-1]:
                # Extract ID from observer
                oid = o.host_ind
                rewards = np.asarray(o.reward)
                self.reward_traces[oid, episode] = np.sum(rewards)

                states = np.asarray(o.rl_observation_vectors) # This should be the chain of observation vectors!!!!!
                actions = np.asarray(o.plan) # This is the chosen indices to navigate to at each timestep

                deltas = self.calculate_deltas(rewards)

                self.policies[oid].update(states, actions, deltas)

        # Return the raw data which we can then investigate and plot to show convergence of policy over time
        return self.mdp.episode_metadata, self.reward_traces

    def calculate_deltas(self, rewards):
        """
        Generate a list of the discounted future rewards at each step of an episode
        Note that discounted_reward[T-2] = rewards[T-1] + discounted_reward[T-1] * gamma.
        We can use that pattern to populate the discounted_rewards array.
        """
        T = len(rewards)
        discounted_future_rewards = np.zeros(T)

        # The final discounted reward is the reward you get at that step
        discounted_future_rewards[-1] = rewards[-1]
        for t in reversed(range(0, T - 1)):
            discounted_future_rewards[t] = (
                rewards[t]
                + discounted_future_rewards[t + 1] * self.mdp.get_discount_factor()
            )
        deltas = (self.mdp.get_discount_factor() ** np.arange(T)) * discounted_future_rewards
        return deltas
    
    def plot_reward_traces(self):
        for i,trace in enumerate(self.reward_traces):
            plt.plot(trace, label=f'Host {i}')
        plt.title('Training Episode Reward Traces')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()


if __name__=="__main__":
    sats = load_satellites()

    import time
    from skyfield.api import load

    # Select a set of hosts and make targets a view of the rest of the stuff in that list of satellites
    hosts = sats[0:2]
    targets = sats[2:] # Technically this is incorrect, as each telescope should look at the other hosts too!!!

    # Get times
    ts = load.timescale()
    tstart = ts.now()
    tend = tstart + timedelta(minutes=5) # Our time window is 5 minutes long to start
    # times = ts.utc(t0.utc_datetime() + np.asarray([timedelta(minutes=x) for x in range(0, 361)])) # 360 minute (6 hour) timeframe

    observers = [Observer(h, i) for i,h in enumerate(hosts)]

    ##############################################################
    # INIT POLICIES AND EXECUTOR
    state_space = RA.size + RA.size + 1
    action_space = RA.size

    mdp = TelescopeMDP(observers, targets, tstart, tend, state_space, action_space, 0.98)
    policies = [DeepNeuralNetworkPolicy(state_space, action_space, 256) for _ in observers]
    dpg = PolicyGradientExecutor(mdp, policies)

    ##############################################################
    # EXECUTE POLICY TRAINING
    results, reward_traces = dpg.execute(100)
    # dpg.plot_reward_traces()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    np.savez(f'results/reward_traces_{timestamp}.npz', reward_traces)

    # Plot rewards!!!
    import matplotlib.pyplot as plt

    ##############################################################
    # EVALUATE ON FUTURE TIME WINDOW!!!!

    # evaltstart = tend + timedelta(hours=4) # Timedelta, which I believe is evaluated as seconds
    # evaltend = evaltstart + timedelta(minutes=5)

    # Set new start and end times on MDP and pass in the frozen, trained policy
    start = time.perf_counter()
    mdp.run_episode(policies)
    elapsed = time.perf_counter() - start

    # Get our results
    evalres = mdp.episode_metadata[0]
    for o in evalres: 
        # print(o.as_dict())
        o.save(f"results/results_{o.host_ind}_{timestamp}.json")
        o.save_maps(f"results/maps_{o.host_ind}_{timestamp}.npz")
        policies[o.host_ind].save(f'results/policy_{o.host_ind}_mdpdpg_{timestamp}.v1.ptm')

    # print(f"Simulated observation planning took {elapsed / 3600:.3f} [h] to complete a plan for {len(hosts)} hosts spanning {evaltstart.utc_iso()} to {evaltend.utc_iso()}.")
    print(f"Simulated observation planning took {elapsed / 3600:.3f} [h] to complete a plan for {len(hosts)} hosts spanning {tstart.utc_iso()} to {tend.utc_iso()}.")
