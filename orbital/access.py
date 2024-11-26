"""
Any and all access opportunity calculators I can muster, plus a spline interpolator for arbitrary stacks of arrays

Primary AO constraints (FOCUS ON THESE SIMPLE GUYS FOR THIS PROJECT!!!!!!!):
    - KOZ (Earth, Moon, Sun, Galactic features (fixed in RA/DEC which is nice...))
    - Range to Target
    - Sunlit (skyfield utility)

Secondary AO constraints (otherwise denoting observation quality degradation)
    - Visual Magnitude (SNR, BRDF, other brightness related measures or thresholds)
    - Relative Velocity (How much is stuff smearing?)
    - Star field density (background radiance / clutter) *OR* simple 'about galactic plane' buffer KOZ
    - ETC

NOTE TO SELF: Fast dot product in numpy of rows of two matrices is multiply then sum (duh)
                def dot(a,b): return np.sum(a*b, axis=1)
"""
import numpy as np
from skyfield.api import load

EARTH_SEMI_MINOR_WGS84 = 6356.752314245 # km
EARTH_SEMI_MAJOR_WGS84 = 6378.137 # km
SUN_KOZ = np.deg2rad(15.0) # Overly conservative, just degrees from center
# EARTH_KOZ_FROM_LIMB = 2.0 # Overly conservative, takes into account angular extent of Earth and adds padding
MOON_KOZ = np.deg2rad(10.0) # Overly conservative, just degrees from center

# Load planets up front!
eph = load('de421.bsp')
EARTH, MOON, SUN = eph['earth'], eph['moon'], eph['sun']

# Vectorized dot product helper
def dot(a, b):
    return np.sum(a * b, axis=1)

# This function came from: https://stephenhartzell.medium.com/satellite-line-of-sight-intersection-with-earth-d786b4a6a9b6
# Modified to be a little bit more vectorized
def los_to_earth(observer_pos, target_pos, padding=90.):
    """Find the intersection of a pointing vector with the Earth
    Finds the intersection of a pointing vector u and starting point s with the WGS-84 geoid
        NOTE: args must be in [km] and must be using a cartesian GCRF ref frame so ellipsoid is evaluated properly 
    Args: 
        observer_pos (np.ndarray): (1x3) array defining the observer position
        target_pos (np.ndarray): (Nx3) array defining the target positions
    Returns:
        np.ndarray of booleans representing whether each target state is behind Earth
    """

    # ADD PADDING PARAMETER TO THE ELLIPSOID TO REPRESENT FIXED KOZ (compensate for earth shine or atmosphere for instance)
    # padding represents where the ionosphere roughly begins @ 90km above sea level
    a = EARTH_SEMI_MAJOR_WGS84 + padding
    b = EARTH_SEMI_MAJOR_WGS84 + padding
    c = EARTH_SEMI_MINOR_WGS84 + padding

    x = observer_pos[0]
    y = observer_pos[1]
    z = observer_pos[2]
    
    # Calculate pointings on the fly
    los = target_pos - observer_pos
    u = los[:,0]
    v = los[:,1]
    w = los[:,2]

    value = -a**2*b**2*w*z - a**2*c**2*v*y - b**2*c**2*u*x
    radical = a**2*b**2*w**2 + a**2*c**2*v**2 - a**2*v**2*z**2 + 2*a**2*v*w*y*z - a**2*w**2*y**2 + b**2*c**2*u**2 - b**2*u**2*z**2 + 2*b**2*u*w*x*z - b**2*w**2*x**2 - c**2*u**2*y**2 + 2*c**2*u*v*x*y - c**2*v**2*x**2
    magnitude = a**2*b**2*w**2 + a**2*c**2*v**2 + b**2*c**2*u**2

    d = (value - a*b*c*np.sqrt(radical)) / magnitude
    return ~np.logical_or(radical < 0, d < 0) # Returns a numpy array of booleans where violations are True

    # if radical < 0:
    #     # raise ValueError("The Line-of-Sight vector does not point toward the Earth")
    #     return False

    # d = (value - a*b*c*np.sqrt(radical)) / magnitude

    # if d < 0:
    #     # raise ValueError("The Line-of-Sight vector does not point toward the Earth")
    #     return False

    # return True
    # return np.array([
    #     x + d * u,
    #     y + d * v,
    #     z + d * w,
    # ])

# Takes a time and spacecraft states at that time
def in_major_keep_out_zones(t, observer, targets):
    # Return True or False (or an array of booleans)
    # This should evaluate Earth, Moon, and Sun KOZ at time t against spacecraft states and return the union (True == 'this state was in KOZ, so throw out')
    # Earth

    observer_pos = observer.at(t).position.km
    target_pos = np.asarray([targ.at(t).position.km for targ in targets])
    violations = np.asarray(los_to_earth(observer_pos, target_pos))

    # Moon
    moon_pos = (MOON - EARTH).at(t).position.km # GCRF
    moon_los = (moon_pos - observer_pos)[None, ...]
    los = target_pos - observer_pos

    obstarg_unit_vec = los/np.linalg.norm(los, axis=1).reshape(los.shape[0],1,los.shape[-1])
    obsmoon_unit_vec = moon_los/np.linalg.norm(moon_los, axis=1)
    sep = np.arccos(dot(obsmoon_unit_vec, obstarg_unit_vec))
    violations = np.logical_or(violations, sep < MOON_KOZ)

    # Sun
    sun_pos = (SUN - EARTH).at(t).position.km # GCRF
    sun_los = (sun_pos - observer_pos)[None, ...]
    obssun_unit_vec = sun_los/np.linalg.norm(sun_los, axis=1)

    sep = np.arccos(dot(obstarg_unit_vec, obssun_unit_vec))
    violations = np.logical_or(violations, sep < SUN_KOZ)

    return violations

# Takes a time and spacecraft states at that time (min and max are km)
def out_of_range(t, observer, targets, min_r=10., max_r=24000.):
    rs = np.asarray([(targ - observer).at(t).distance().km for targ in targets])
    return np.logical_or(rs < min_r, rs > max_r) # True denotes violations

# Skyfield method
def not_sunlit(t, targets):
    # Not sure if this syntax will work, might need to evaluate per satellite (target) in targets at all times t for efficiency
    return ~np.asarray([targ.at(t).is_sunlit(eph) for targ in targets]) # True denotes violations

# Skyfield method
def behind_earth(t, observer, targets):
    # Not sure how fast this is...
    return np.asarray([(observer).at(t).observe(targ).apparent().is_behind_earth() for targ in targets])


if __name__=="__main__":
    # Test script for loading satellites and applying access constraints prior to density calculation
    from density import *

    # Satellites
    sats = load_satellites()

    # Compute 