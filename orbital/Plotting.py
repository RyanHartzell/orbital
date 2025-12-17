# Plotting for results
import numpy as np
import matplotlib.pyplot as plt

class ObserverData:
    def __init__(self, npz_path):
        self.path = npz_path
        self._load()

    def _load(self):
        data = np.load(self.path, allow_pickle=True)

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
            "file": self.path,
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


obs = ObserverData("results/Dec_MCTS/observer_0_arrays.npz")

print(obs.summary())

# Plot belief map at all time index
for t in range(len(obs.local_belief)):
    plt.imshow(obs.local_belief_map[t], cmap="inferno")
    plt.title(f"Observer 0 Local Belief Map @ t={t}")
    plt.colorbar()
    plt.show()
