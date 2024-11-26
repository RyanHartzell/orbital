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

# THESE SHOULD BE MOVED INTO A DENSITY CLASS HOLY SHIT
SCALE = 1
RA,DEC = np.meshgrid(np.linspace(0,2*np.pi,int(360*SCALE+1)), np.linspace(-np.pi/2, np.pi/2, int(180*SCALE+1)))

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
def calculate_apparent_radecrange(host, targets, times, mask=None):
    # By default ICRS. Note that to account for light-time we should technically use the observe function.... like satellite.at(t).observe(target).apparent().radec()
    # Will need to call .radians and .km on each of the return elements in order to get numpy arrays

    if mask is not None:
        res = []
        for i,t in enumerate(times):
            res.append([(targ - host).at(t).radec() for targ in targets[mask[:,i]]])
        return res
    else:
        return [(t - host).at(times).radec() for t in targets]

# Return contiguous numpy array of values for ra, dec, and ranges separately
def reformat_radecrange(apparent_radecrange, ragged=False):
    if ragged:
        ras = []
        decs = []
        ranges = []
        for t in apparent_radecrange: # stored by time
            tra = []
            tdec = []
            trng = []
            for targ in t:
                tra.append(targ[0].radians)
                tdec.append(targ[1].radians)
                trng.append(targ[2].km)
            ras.append(tra)
            decs.append(tdec)
            ranges.append(trng)

        return (ras, decs, ranges) # This is due to different length sub-vectors

    else:
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

# This guy just takes 1d vectors for ra and dec values
def construct_ball_tree(ras, decs):
    if len(ras) > 0:
        return BallTree(np.c_[decs, ras], metric='haversine')
    else:
        return None

def construct_kde_map(bt, afov=5.5, **kwargs):
    if bt is None:
        return np.zeros_like(RA)
    return bt.kernel_density(np.c_[DEC.flat, RA.flat], np.deg2rad(afov/2), **kwargs).reshape(RA.shape)

# afov is full size of largest dimension of aperture in degrees
def construct_fov_density_map(bt, afov=5.5):
    if bt is None:
        # (Density PDF , query_result_grid) 
        # Density can't be zero as we need to always be able to randomly sample it!!!
        return (np.ones_like(RA)/RA.size, [np.empty(0) for _ in range(RA.size)])

    # Get list of numpy arrays of indices at each query point (basically the flattened index of RA or DEC)
    query_results = bt.query_radius(np.c_[DEC.flat, RA.flat], np.deg2rad(afov/2)) # list of numpy arrays I believe?
    counts = np.array([len(q) for q in query_results]).reshape(RA.shape)
    return counts / np.sum(counts), query_results

# Times should be skyfield or astropy times with a utc_iso() method for formatting
def animate_heatmaps(heatmaps, times, to_disk=False, filename='test'):
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
        with open(filename+'.html', 'w') as f:
            f.write(doc)
    return doc

if __name__=="__main__":
    # Basic flow
    sats = load_satellites()

    # # Select a host and make targets a view of the rest of the stuff in that list of satellites
    host = sats[99]
    targets = sats[:99] + sats[100:]

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

    # # Calculate apparent ra, dec, ranges relative to host state at each time t
    obs = reformat_radecrange(calculate_apparent_radecrange(host, np.asarray(targets), times, access), ragged=True)

    # # Build all ball trees
    # bts = construct_ball_trees(obs)
    bts = [construct_ball_tree(obs[0][i], obs[1][i]) for i in range(len(times))]

    # # Build all density maps
    # kde_maps = np.dstack([construct_kde_map(bt) for bt in bts]) 
    explicit_maps = np.dstack([construct_fov_density_map(bt)[0] for bt in bts]) # Be sure to only get the density map and not the actual indices returned from the query

    # # Make animation of density maps!
    # doc = animate_heatmaps(kde_maps, times, True, 'test2')
    doc = animate_heatmaps(explicit_maps, times, True, 'test2_explicit')
    # print(len(doc))