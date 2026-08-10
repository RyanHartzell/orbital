# Plotting for results
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
from density import *

def compute_uncertainty_map():
    # This should be our global (implicit) objective at a given timestep, a given host, and target records
    pass

def plot_total_catalog_uncertainty_over_time():
    # This should be able to gauge performance of greedy vs Dec-MCTS in reducing catalog uncertainty over time

    # Use target records, access, and density (query results) for all times in plan
    pass

def compute_value_map():
    # We'll probably average these in some window and plot plan on top (ala figure in paper)
    pass

def plot_access_by_time(host, times, access):
    # print(access.shape)

    # Simple accessibility plot
    fig = plt.figure()
    plt.tick_params(axis='both', color='k', labelcolor='k')
    # plt.plot(times.utc_datetime(), access.sum(0), label='Total Access')
    plt.plot(times, np.squeeze(access).sum(1) / (np.min(np.squeeze(access).shape)), label='Total Access')
    plt.title(f"Valid Access Opportunities T+0.5 [hr]\n{host}", color='k')
    plt.xlabel("Time", color='k')
    plt.ylabel(r"% Targets Accessible", color='k')
    plt.legend(loc='upper right')
    plt.show()

def plot_access_by_target(host, targets, access):
    # Simple accessibility plot
    fig = plt.figure()
    plt.tick_params(axis='both', color='k', labelcolor='k')
    # plt.plot(times.utc_datetime(), access.sum(0), label='Total Access')        
    plt.plot(targets, np.squeeze(access).sum(0) / (np.max(np.squeeze(access).shape)), label=r'Access % by Target')
    plt.title(f"Per Target Access Opportunities T+0.5 [hr]\n{host}", color='k')
    plt.xlabel("Time", color='k')
    plt.ylabel(r"Target Accessibility", color='k')
    plt.legend(loc='upper right')
    plt.show()

def compute_observation_stats(times, plan):
    # Plot # observations per target for a given plan (or across all plans interleaved?)

    # Access normalized targeting histogram (# of times target was viewed / % of time accessible across observers)

    # % Coverage -> % targets viewed at least once per 30min
    pass

def belief_update_vis():
    # 
    pass

class ObserverData:
    def __init__(self, json_path, npz_path):
        self.json_path = json_path
        self.npz_path = npz_path
        self._load()

    def _load(self):
        # JSON first
        with open(self.json_path, 'r') as f:
            self.results = json.load(f) # Should load into dictionary...

        # NPZ next
        data = np.load(self.npz_path, allow_pickle=True)

        self.local_belief = data["local_belief"]
        self.extern_belief = data["extern_belief"]

        self.local_belief_map = list(data["local_belief_map"])
        self.extern_belief_map = list(data["extern_belief_map"])

        self.density_maps = list(data["density_maps"])
        self.access = list(data["access"])

        self.num_times = self.local_belief.shape[0]
        self.num_targets = self.local_belief.shape[1]

    def summary(self):
        return {
            "num_times": self.num_times,
            "num_targets": self.num_targets,
            "map_shape": self.local_belief_map[0].shape,
        }

    def get_time_slice(self, t_idx):
        return {
            "local_belief": self.local_belief[t_idx],
            "extern_belief": self.extern_belief[t_idx],
            "local_map": self.local_belief_map[t_idx],
            "extern_map": self.extern_belief_map[t_idx],
            "density_map": self.density_maps[t_idx],
            "access": self.access[t_idx],
        }

if __name__=="__main__":


    # obs = ObserverData(sys.argv[1])

    # print(obs.summary())

    # plt.ion()
    # fig, axes = plt.subplots(2)
    # plt.tight_layout()
    # cbar1 = None
    # cbar2 = None
    # # Plot belief map at all time index
    # for t in range(len(obs.local_belief)):
    #     a1 = axes[0].imshow(obs.local_belief_map[t], cmap="inferno", vmax=obs.local_belief_map[t].max()*0.75)
    #     axes[0].set_title(f"Observer Local Belief Map @ t={t}")

    #     a2 = axes[1].imshow(obs.extern_belief_map[t], cmap="inferno", vmax=obs.extern_belief_map[t].max()*0.75)
    #     axes[1].set_title(f"Observer External Belief Map @ t={t}")

    #     if cbar1 is None or cbar2 is None:
    #         cbar1 = fig.colorbar(a1, ax=axes[0], fraction=0.046, pad=0.04)
    #         cbar2 = fig.colorbar(a2, ax=axes[1], fraction=0.046, pad=0.04)
    #     plt.draw()
    #     plt.pause(0.2)
    #     for ax in axes.flat:
    #         ax.cla()

    # plt.close()
    # plt.ioff()

    # # Plot average belief and plan as green->white markers?
    # plt.imshow(np.mean(obs.density_maps, axis=0), cmap='inferno', norm='symlog')
    # plt.title("Avg Density")
    # plt.show()

    ################################################################################
    # Read all observers data:
    import glob
    d = sys.argv[1]
    print("Received directory for processing: ", d)

    obs_list = [ObserverData(i,j) for i,j in zip(sorted(glob.glob(d+"*.json")), sorted(glob.glob(d+"*.npz")))]
    N = obs_list[0].num_times
    M = obs_list[0].num_targets

    # Per target accessibility
    fused = np.dstack([np.asarray(o.access) for o in obs_list]).mean(-1) # Elementwise sum gives count of target obs at every time across observers, sum along large dim gives us per target availability % when divided by len(time)
    print(fused.shape)
    plt.imshow(fused, aspect_ratio='auto', cmap='magma')

    # Plot access example plot
    plot_access_by_time("All Hosts", range(N), fused)
    plot_access_by_target("All Hosts", range(M), fused)

    # Now generate plot of 
    #compute_observation_stats(range(N), fused)
