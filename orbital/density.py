"""
Calculate AzEl from ephemeris data, then density
"""
import httpx
import json
from skyfield.api import load, EarthSatellite
from sklearn.neighbors import BallTree

class Frame:
    def __init__(self) -> None:
        pass

class PerspectiveCamera:
    def __init__(self) -> None:
        pass

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
    pass

# Construct BallTree for each timestep
# Requires 'transposing' the results from the calculate_apparent_radecrange function so we have all ra/dec for all targets grouped per 
def construct_ball_tree():
    pass

def construct_kde_map():
    pass

# afov is full size of largest dimension of aperture in degrees
def construct_fov_density_map(afov=5.5):
    pass

if __name__=="__main__":
    pass