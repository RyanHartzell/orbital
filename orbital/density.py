"""
Calculate AzEl from ephemeris data, then density
"""
import httpx
import json
import numpy as np
from skyfield.api import load, EarthSatellite, Time
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

__all__=(
    "SCALE",
    "RA",
    "DEC",
    "load_satellites",
    "make_ephemerides",
    "calculate_relative_states",
    "calculate_apparent_radecrange",
    "reformat_radecrange",
    "construct_ball_trees",
    "construct_ball_tree",
    "construct_kde_map",
    "construct_fov_density_map",
    "animate_heatmaps",
    "animate_observation_plan",
    "animate_all_observation_plans",
)

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
    fig, ax = plt.subplots()
    sns.set_style('darkgrid')
    ax.set_title(f"Host-Centered Apparent Geocentric ICRF \nTarget Density at $t=${times[0].utc_iso() if isinstance(times[0], Time) else times[0]}", color='black')
    ax.set_ylabel(r"$\alpha$ (Right Ascension) [rad]", color='black')
    ax.set_xlabel(r"$\delta$ (Declination) [rad]", color='black')

    ax.imshow(heatmaps[:,:,0], cmap="inferno", extent=(0, 2*np.pi, -np.pi/2, np.pi/2), origin='lower', vmax=heatmaps.max())

    # Define the animation function
    def update(frame):
        ax.cla()
        sns.set_style('darkgrid')
        ax.set_title(f"Host-Centered Apparent Geocentric ICRF \nTarget Density at $t=${times[frame].utc_iso() if isinstance(times[0], Time) else times[0]}", color='black')
        ax.set_ylabel(r"$\alpha$ (Right Ascension) [rad]", color='black')
        ax.set_xlabel(r"$\delta$ (Declination) [rad]", color='black')
        ax.imshow(heatmaps[:,:,frame], cmap="inferno", extent=(0, 2*np.pi, -np.pi/2, np.pi/2), origin='lower', vmax=heatmaps.max())

        return ax

    # Create the animation object
    ani = FuncAnimation(fig, update, frames=list(range(1,len(times)))) #, interval=20, blit=True)
    doc = ani.to_jshtml()
    if to_disk:
        with open(filename+'.html', 'w') as f:
            f.write(doc)
    return doc

# Times should be skyfield or astropy times with a utc_iso() method for formatting, or string
def animate_observation_plan(heatmaps, obs_ra, obs_dec, times, to_disk=False, filename='test'):
    # print("Heatmaps shape: ", heatmaps.shape)
    fig, ax = plt.subplots()
    # sns.set_style('darkgrid')
    ax.grid(False)
    ax.set_title(f"Host-Centered Apparent Geocentric ICRF \nTarget Density at $t=${times[0].utc_iso() if isinstance(times[0], Time) else times[0]}", color='black')
    ax.set_ylabel(r"$\alpha$ (Right Ascension) [rad]", color='black')
    ax.set_xlabel(r"$\delta$ (Declination) [rad]", color='black')

    ax.imshow(heatmaps[:,:,0], cmap="inferno", extent=(0, 2*np.pi, -np.pi/2, np.pi/2), origin='lower', vmax=heatmaps.max())

    # Plot obs position
    ax.scatter(obs_ra[0], obs_dec[0],s=20*4**2, c='green', marker='1', capstyle='round')

    # Define the animation function
    def update(frame):
        ax.cla()
        # sns.set_style('darkgrid')
        ax.grid(False)
        ax.set_title(f"Host-Centered Apparent Geocentric ICRF \nTarget Density at $t=${times[frame].utc_iso() if isinstance(times[0], Time) else times[0]}", color='black')
        ax.set_ylabel(r"$\alpha$ (Right Ascension) [rad]", color='black')
        ax.set_xlabel(r"$\delta$ (Declination) [rad]", color='black')
        ax.imshow(heatmaps[:,:,frame], cmap="inferno", extent=(0, 2*np.pi, -np.pi/2, np.pi/2), origin='lower', vmax=heatmaps.max())

        # Add observation point
        ax.scatter(obs_ra[frame], obs_dec[frame],s=20*4**2, c='green', marker='1', capstyle='round') #, markeredgecolor='white')

        return ax

    # Create the animation object
    ani = FuncAnimation(fig, update, frames=list(range(1,len(times)))) #, interval=20, blit=True)
    doc = ani.to_jshtml()
    if to_disk:
        with open(filename+'.html', 'w') as f:
            f.write(doc)
    return doc

# Times should be skyfield or astropy times with a utc_iso() method for formatting, or string
# This should plot ALL obs plans as time-ordered where the obs are colored by their respective host id (categorical)
# Okay....... this should probably be a bunch of subplots tbh..................
def animate_all_observation_plans(heatmaps, obs_ra, obs_dec, times, ids, to_disk=False, filename='test'):

    # TODO: Move merging and sorting code in here.....
    import matplotlib as mpl

    colors = [mpl.color_sequences['tab10'][i%10] for i in range(max(ids)+1)]
    print(len(colors))

    fig, ax = plt.subplots()
    # sns.set_style('darkgrid')
    ax.grid(False)
    ax.set_title(f"Host-Centered Apparent Geocentric ICRF \nTarget Density at $t=${times[0].utc_iso() if isinstance(times[0], Time) else times[0]}", color='black')
    ax.set_ylabel(r"$\alpha$ (Right Ascension) [rad]", color='black')
    ax.set_xlabel(r"$\delta$ (Declination) [rad]", color='black')

    ax.imshow(heatmaps[:,:,0], cmap="inferno", extent=(0, 2*np.pi, -np.pi/2, np.pi/2), origin='lower', vmax=heatmaps.max())

    # Plot obs position
    ax.scatter(obs_ra[0], obs_dec[0],s=20*4**2, color=colors[ids[0]], marker='1', capstyle='round')

    # Define the animation function
    def update(frame):
        ax.cla()
        # sns.set_style('darkgrid')
        ax.grid(False)
        ax.set_title(f"Host-Centered Apparent Geocentric ICRF \nTarget Density at $t=${times[frame].utc_iso() if isinstance(times[0], Time) else times[frame]}", color='black')
        ax.set_ylabel(r"$\alpha$ (Right Ascension) [rad]", color='black')
        ax.set_xlabel(r"$\delta$ (Declination) [rad]", color='black')
        ax.imshow(heatmaps[:,:,frame], cmap="inferno", extent=(0, 2*np.pi, -np.pi/2, np.pi/2), origin='lower', vmax=heatmaps.max())

        # Add observation point
        ax.scatter(obs_ra[frame], obs_dec[frame],s=20*4**2, color=colors[ids[frame]], marker='1', capstyle='round') #, markeredgecolor='white')

        return ax

    # Create the animation object
    ani = FuncAnimation(fig, update, frames=list(range(1,len(times)))) #, interval=20, blit=True)
    doc = ani.to_jshtml()
    if to_disk:
        with open(filename+'.html', 'w') as f:
            f.write(doc)
    return doc


if __name__=="__main__":
    # # Basic flow
    # sats = load_satellites()

    # # # Select a host and make targets a view of the rest of the stuff in that list of satellites
    # host = sats[99]
    # targets = sats[:99] + sats[100:]

    # # Get times
    # from datetime import timedelta
    # ts = load.timescale()
    # t0 = ts.now()
    # times = ts.utc(t0.utc_datetime() + np.asarray([timedelta(minutes=x) for x in range(0, 361)])) # 360 minute (6 hour) timeframe

    # # Get access mask
    # from access import in_major_keep_out_zones, not_sunlit, out_of_range
    # sunlit_access = not_sunlit(times, targets)
    # print(f"% access [SUNLIT] = {np.sum(~sunlit_access)/sunlit_access.size * 100.}")

    # range_access = out_of_range(times, host, targets)
    # print(f"% access [IN-RANGE] = {np.sum(~range_access)/range_access.size * 100.}")

    # koz_access = in_major_keep_out_zones(times, host, targets)
    # print(f"% access [NOT-IN-KOZ] = {np.sum(~koz_access)/koz_access.size * 100.}")

    # # Construct overall access mask (should be SATNUM x TIMESTEP)
    # access = ~sunlit_access * ~range_access * ~koz_access # We can multiply these since any zero value should cause a switch to False
    # print(f"Total % access across timesteps = {np.sum(access)/access.size * 100.}")

    # # Plot access over time as total satellites available for observation at each timestep
    # plt.plot(times.utc_datetime(), access.sum(0))
    # plt.title("Access Plot (# of observable targets)")
    # plt.show()

    # # Calculate apparent ra, dec, ranges relative to host state at each time t
    # obs = reformat_radecrange(calculate_apparent_radecrange(host, np.asarray(targets), times, access), ragged=True)

    # # Build all ball trees
    # bts = construct_ball_trees(obs)
    # bts = [construct_ball_tree(obs[0][i], obs[1][i]) for i in range(len(times))]

    # # Build all density maps
    # kde_maps = np.dstack([construct_kde_map(bt) for bt in bts]) 
    # explicit_maps = np.dstack([construct_fov_density_map(bt)[0] for bt in bts]) # Be sure to only get the density map and not the actual indices returned from the query

    # # Make animation of density maps!
    # doc = animate_heatmaps(kde_maps, times, True, 'test2')
    # doc = animate_heatmaps(explicit_maps, times, True, 'test2_explicit')
    # print(len(doc))

    ######################################################
    # TEST OBS PLAN VISUALIZATION!!! (Should be moved into its own file probably...)

    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import glob

    # This is our training plot of reward traces per training episode (aka 100 rewards for 100 episodes)
    for fr in glob.glob('results/reward_traces_*-15.npz'):
        training_rewards = np.load(fr)['arr_0']
        print(training_rewards.shape) # should be (# hosts, # episodes)

        plt.plot(training_rewards.T)
        plt.legend(labels=range(training_rewards.shape[0]))
        plt.show()

    # These are the evaluation observer records, so a single episode
    for fj, fhm in zip(glob.glob('results/*-15.json'), glob.glob('results/*-15.npz')):        
        observer_record = json.load(open(fj))
        obs_inds = observer_record['Plan']
        obs_ra = RA.flat[obs_inds]
        obs_dec = DEC.flat[obs_inds]

        value = np.load(fhm)['arr_0']
        animate_observation_plan(value, obs_ra, obs_dec, observer_record['EndTimes'], to_disk=True, filename=f"{fhm.split('.')[0]}")

    # import glob
    # jsons = glob.glob('results/*.json')
    # heatmaps = glob.glob('results/*.npz') 

    # def load_heatmap(files):            
    #     for f in files:
    #         yield np.load(f)['arr_0'] 

    # def load_obs_rec(files):
    #     for f in files:
    #         with open(f) as j:       
    #             yield json.load(j)

    # recs = [*load_obs_rec(jsons)]  
    # hm = [*load_heatmap(heatmaps)]

    # new_times = sum([r['EndTimes'] for r in recs], []) 
    # timesort = np.argsort(new_times) 
    # hmstack = np.dstack(hm) 
    # hmsort = hmstack[..., timesort] 
    # print(hmsort.shape)

    # plans = sum([r['Plan'] for r in recs], []) 
    # plansort = np.asarray(plans)[timesort] 
    # full_obs_ra = RA.flat[plansort] 
    # full_obs_dec = DEC.flat[plansort] 
    # print(len(full_obs_dec))

    # ids = sum([[r['Index']]*len(r['Plan']) for r in recs], []) 
    # ids = np.asarray(ids)[timesort]
    # print(len(ids))

    # timesorted = np.asarray(new_times)[timesort]
    # print(len(timesorted))

    # # Now try our new func for visualizing all of the obs
    # animate_all_observation_plans(hmsort, full_obs_ra, full_obs_dec, timesorted, ids, to_disk=True, filename='allobsplanviz')