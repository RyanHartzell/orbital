"""
Calculate AzEl from ephemeris data, then density
"""
import httpx
import json
import numpy as np
from skyfield.api import load, EarthSatellite
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# class Frame:
#     def __init__(self) -> None:
#         pass

# class PerspectiveCamera:
#     def __init__(self) -> None:
#         pass

# Skyfield related stuff which we'll use as a basis for the new app (no more poliastro!!!)
def load_satellites():
    r = httpx.get("https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json")
    data = json.loads(r.content)
    ts = load.timescale()
    sats = [EarthSatellite.from_omm(ts, fields) for fields in data]
    print('Loaded', len(sats), 'satellites')

    return sats

def make_ephemerides(sats, times):
    return sats.at(times) # Geocentric ICRS

# This should be the same as Geocentric ICRS (ECI) -> It's an inertial frame locked to the ICRF but centered at Earth at time t
def calculate_relative_states(host, targets, times):
    from skyfield.framelib import ecliptic_J2000_frame
    return [(t - host).at(times).frame_xyz_and_velocity(ecliptic_J2000_frame) for t in targets] # These are positions and velocities relative to host

# This should be simpler than dealing with reference frame of camera and gimbal and whatnot
# Instead we're just using apparent RA/Dec and range and we can use that to build our density map. It's effectively the same as AzEl without adjusting to RIC basically
def calculate_apparent_radecrange(host, targets, times):
    # By default ICRS. Note that to account for light-time we should technically use the observe function.... like satellite.at(t).observe(target).apparent().radec()
    # Will need to call .radians and .km on each of the return elements in order to get numpy arrays
    return [(t - host).at(times).radec() for t in targets]

# Return contiguous numpy array of values for ra, dec, and ranges separately
def reformat_radecrange(apparent_radecrange):
    ras = []
    decs = []
    ranges = []
    for x in apparent_radecrange:
        ras.append(x[0].radians)
        decs.append(x[1].radians)
        ranges.append(x[2].km)

    # targets are rows, time is columns, ra/dec/range is along channels
    return np.dstack([ras, decs, ranges])

# Construct BallTree for each timestep
# Requires 'transposing' the results from the calculate_apparent_radecrange function so we have all ra/dec for all targets grouped per 
def construct_ball_trees(data):
    return [BallTree(np.c_[data[:,t,1], data[:,t,0]], metric='haversine') for t in range(data.shape[1])]

def construct_kde_map(bt, scale=1, afov=5.5, **kwargs):
    RA,DEC = np.meshgrid(np.linspace(0,2*np.pi,360*scale+1), np.linspace(-np.pi/2, np.pi/2, 180*scale+1))
    return bt.kernel_density(np.c_[DEC.flat, RA.flat], np.deg2rad(afov/2), **kwargs).reshape(RA.shape)

# afov is full size of largest dimension of aperture in degrees
def construct_fov_density_map(afov=5.5, scale=(4,4)):
    pass

# Times should be skyfield or astropy times with a utc_iso() method for formatting
def animate_heatmaps(heatmaps, times, to_disk=False):
    print("Heatmaps shape: ", heatmaps.shape)
    fig, ax = plt.subplots()
    ax.set_title(f"Host-Centered Apparent Geocentric ICRF \nTarget Density at $t=${times[0].utc_iso()}")
    ax.set_xlabel(r"$\alpha$ (Right Ascension) [rad]")
    ax.set_xlabel(r"$\delta$ (Declination) [rad]")

    hm = ax.imshow(heatmaps[:,:,0], cmap="inferno")
    # hm.set_extent((0, 2*np.pi, -np.pi/2, np.pi/2))

    # Define the animation function
    def update(frame):
        ax.cla()
        ax.set_title(f"Host-Centered Apparent Geocentric ICRF \nTarget Density at $t=${times[frame].utc_iso()}")
        ax.set_xlabel(r"$\alpha$ (Right Ascension) [rad]")
        ax.set_xlabel(r"$\delta$ (Declination) [rad]")
        hm = ax.imshow(heatmaps[:,:,frame], cmap="inferno")
        # hm.set_extent((0, 2*np.pi, -np.pi/2, np.pi/2))
        return hm

    # Create the animation object
    ani = FuncAnimation(fig, update, frames=list(range(1,len(times)))) #, interval=20, blit=True)
    doc = ani.to_jshtml()
    if to_disk:
        with open('ani.html', 'w') as f:
            f.write(doc)
    return doc

if __name__=="__main__":
    # Basic flow
    sats = load_satellites()

    # Select a host and make targets a view of the rest of the stuff in that list of satellites
    host = sats[0]
    targets = sats[1:]

    # Get times
    ts = load.timescale()
    times = ts.utc(2024, 11, 7, 0, range(0,3*60+1,5)) # hourly

    # Calculate apparent ra, dec, ranges relative to host state at each time t
    obs = reformat_radecrange(calculate_apparent_radecrange(host, targets, times))

    # Build all ball trees
    bts = construct_ball_trees(obs)

    # Build all density maps
    kde_maps = np.dstack([construct_kde_map(bt) for bt in bts])

    # Make animation of density maps!
    doc = animate_heatmaps(kde_maps, times, True)
    print(len(doc))