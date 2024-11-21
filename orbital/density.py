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
import seaborn as sns

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

# This should be (basically) the same as Geocentric ICRS (ECI) -> It's an inertial frame locked to the ICRF but centered at Earth at time t
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
    sns.set_style('darkgrid')
    ax.set_title(f"Host-Centered Apparent Geocentric ICRF \nTarget Density at $t=${times[0].utc_iso()}", color='black')
    ax.set_ylabel(r"$\alpha$ (Right Ascension) [rad]", color='black')
    ax.set_xlabel(r"$\delta$ (Declination) [rad]", color='black')

    hm = ax.imshow(heatmaps[:,:,0], cmap="inferno")
    ax.set_ylim(-np.pi/2, np.pi/2)
    hm.set_extent((0, 2*np.pi, -np.pi/2, np.pi/2))

    # Define the animation function
    def update(frame):
        ax.cla()
        sns.set_style('darkgrid')
        ax.set_title(f"Host-Centered Apparent Geocentric ICRF \nTarget Density at $t=${times[frame].utc_iso()}", color='black')
        ax.set_ylabel(r"$\alpha$ (Right Ascension) [rad]", color='black')
        ax.set_xlabel(r"$\delta$ (Declination) [rad]", color='black')
        hm = ax.imshow(heatmaps[:,:,frame], cmap="inferno")
        ax.set_ylim(-np.pi/2, np.pi/2)
        hm.set_extent((0, 2*np.pi, -np.pi/2, np.pi/2))
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

    # # Select a host and make targets a view of the rest of the stuff in that list of satellites
    host = sats[0]
    targets = sats[1:]

    # Get times
    from datetime import timedelta
    ts = load.timescale()
    t0 = ts.now()
    times = ts.utc(t0.utc_datetime() + np.asarray([timedelta(minutes=x) for x in range(0, 361)])) # 360 minute (6 hour) timeframe

    # Get access mask
    from access import in_major_keep_out_zones, not_sunlit, out_of_range
    sunlit_access = not_sunlit(times, targets)
    print(f"% access [SUNLIT] = {np.sum(~sunlit_access)/sunlit_access.size * 100.}")

    range_access = out_of_range(times, host, targets)
    print(f"% access [IN-RANGE] = {np.sum(~range_access)/range_access.size * 100.}")

    koz_access = in_major_keep_out_zones(times, host, targets)
    print(f"% access [NOT-IN-KOZ] = {np.sum(~koz_access)/koz_access.size * 100.}")

    # Construct overall access mask (should be SATNUM x TIMESTEP)
    access = ~sunlit_access * ~range_access * ~koz_access # We can multiply these since any zero value should cause a switch to False
    print(f"Total % access across timesteps = {np.sum(access)/access.size * 100.}")

    # Plot access over time as total satellites available for observation at each timestep
    plt.plot(times.utc_datetime(), access.sum(0))
    plt.title("Access Plot (# of observable targets)")
    plt.show()

    # Now we filter targets which are accessible at each time for our density maps (ragged until density computed)


    # # Calculate apparent ra, dec, ranges relative to host state at each time t
    # obs = reformat_radecrange(calculate_apparent_radecrange(host, targets, times))

    # # Build all ball trees
    # bts = construct_ball_trees(obs)

    # # Build all density maps
    # kde_maps = np.dstack([construct_kde_map(bt) for bt in bts])

    # # Make animation of density maps!
    # doc = animate_heatmaps(kde_maps, times, True)
    # print(len(doc))