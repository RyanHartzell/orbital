"""
Created on Fri Mar  9 16:22:46 2018
Modified on Thu Apr 4 2024

@original-author:   bonar
@modified-by:       RyanHartzell
"""

import matplotlib.pyplot as plt
import numpy as np

class Orbit:
    pass
class StateVector:
    def __init__(self, t, s):
        self.time = t
        self.data = s
    @property
    def state(self):
        return self.data
    @property
    def t(self):
        return self.time
    @property
    def pos(self):
        return self.data[:3]
    @pos.setter
    def pos(self, p):
        self.data[:3] = p
    @property
    def vel(self):
        return self.data[3:]
    @vel.setter
    def vel(self, v):
        self.data[3:] = v

    # This allows us to step by a variable amount. State transition matrix is handled externally
    def __call__(self, stm):
        self.data = stm @ self.data.T

class IManeuver:
    pass
class ImpulsiveManeuver(IManeuver):
    pass
class ImpulsiveManeuverSeries:
    pass
class Controller:
    pass
class Spacecraft:
    pass
class IConstraint:
    pass
class PhysicalConstraint(IConstraint):
    pass
class OperationalConstraint(IConstraint):
    pass

# Hohmann transfer function?
def hohmann(r0, rdot0, rf, rdotf):
    return

# given estimated destination state, generate MINIMUM (lowest cost!!) delta V impulses needed to reach this point
def simple_ideal_controller(state0, statef, total_dv):
    impulses = [0.0]
    return impulses

def simple_maneuver_feasibility(state0, statef, max_dv, max_dt):
    return True # returns true or false given whether the maneuver can be completed within the constraints

# Do everything in kilometers for ease
r0 = [0, 0, 0]                  #initial starting position
rdot0 = [0.2, 0.1, -0.15]            #initial velocity (on release from ISS) (m/s)
R = 405 + 6870   # Altitude from center of earth in km
mu = 398600.5 # G (M + m) aka standard gravitational parameter for Earth centered system (in m^3 / s^2)
omeg = np.sqrt(mu/R**3) # orbital rate in radians per second

# Convert this into a maneuver injection function (basically add a random delta-v impulse or series therein during course of animation)
nframes = 1000
nframes2 = 200
nframes3 = 200
dt = 20

def CW2(r0, rdot0, omeg, t):
    x0 = r0[0]
    y0 = r0[1]
    z0 = r0[2]
    xdot0 = rdot0[0]
    ydot0 = rdot0[1]
    zdot0 = rdot0[2]

    xt = (4*x0 + (2*ydot0)/omeg)+(xdot0/omeg)*np.sin(omeg*t)-(3*x0+(2*ydot0)/omeg)*np.cos(omeg*t)
    yt = (y0 - (2*xdot0)/omeg)+((2*xdot0)/omeg)*np.cos(omeg*t)+(6*x0 + (4*ydot0)/omeg)*np.sin(omeg*t)-(6*omeg*x0+3*ydot0)*t
    zt = z0*np.cos(omeg*t)+(zdot0/omeg)*np.sin(omeg*t)
    
    xdott = (3*omeg*x0+2*ydot0)*np.sin(omeg*t)+xdot0*np.cos(omeg*t)
    ydott = (6*omeg*x0+4*ydot0)*np.cos(omeg*t)-2*xdot0*np.sin(omeg*t)-(6*omeg*x0+3*ydot0)
    zdott = zdot0*np.cos(omeg*t)-z0*omeg*np.sin(omeg*t)
    
    return([xt,yt,zt],[xdott,ydott,zdott])

def CW_vectorized_single(state, w, t, i):
    # state is [r0, v0]
    # transition matrix instead of functional evaluation:
    state = np.asarray(state)
    wt = w * t
    recw = 1 / w
    swt = np.sin(wt)
    cwt = np.cos(wt)
    mcwt = 1 - cwt

    # Throw if there are bad values 
    for v in [state, wt, recw]:
        try:
            np.asarray_chkfinite([v])
        except:
            print("[ERROR] Non-finite values in: ", v)
            raise


    M = np.array([[4 - 3*cwt, 0, 0, recw*swt, 2*recw*mcwt, 0],
                   [6*(swt - wt), 1, 0, -2*recw*mcwt, recw*(4*swt-3*wt), 0],
                   [0, 0, cwt, 0, 0, recw*swt],
                   [3*swt, 0, 0, cwt, 2*swt, 0],
                   [-6*recw*mcwt, 0, 0, -2*swt, 4*cwt-3, 0],
                   [0, 0, -w*swt, 0, 0, cwt]])

    # I believe this would also work for a stack of matrices...
    # state.T[:] = M @ state.T
    return M @ state.T
    # return state @ M

# Seems to be working!!!! :D
def CW_vectorized_batch(state, w, t):
    state = np.asarray(state)
    t = np.atleast_1d(t)
    wt = w * t
    recw = 1 / w
    swt = np.sin(wt)
    cwt = np.cos(wt)
    mcwt = 1 - cwt
    z = np.zeros_like(t)
    o = np.ones_like(t)

    # # Throw if there are bad values
    # try: 
    #     np.asarray_chkfinite([state, wt, recw, swt, cwt, mcwt])
    # except ValueError:
    #     raise

    M = np.transpose(np.array([[4 - 3*cwt, z, z, recw*swt, 2*recw*mcwt, z],
                   [6*(swt - wt), o, z, -2*recw*mcwt, recw*(4*swt-3*wt), z],
                   [z, z, cwt, z, z, recw*swt],
                   [3*w*swt, z, z, cwt, 2*swt, z],
                   [-6*w*mcwt, z, z, -2*swt, 4*cwt-3, z],
                   [z, z, -w*swt, z, z, cwt]]), (2,0,1))

    # I believe this would also work for a stack of matrices...
    return M @ state.T
    # return state @ M

# Display
def plot_cw(states):
    ax = plt.subplot(projection='3d')
    ax.set_xlabel('X`')
    ax.set_ylabel('Y`')
    ax.set_zlabel('Z`')
    ax.scatter(states[:,0], states[:,1], states[:,2])
    plt.show()

def convert_ECI_to_Hill():
    return

def convert_ECI_to_HCW_to_ROE():
    return

def calculate_states(initrv, omeg, times):
    return CW_vectorized_batch(initrv, omeg, times)

# TEST MODULES
def test_compare_states(*args):
    initrv = [100,50,20,0.0005,-0.001,0.003]
    states1 = calculate_states(initrv, *args)
    states2 = calculate_states(initrv, *args)

    print(testvar := np.allclose(states1, states2))
    assert(testvar)

if __name__=="__main__":
    # # Success
    # test_compare_states(omeg, np.linspace(1,5000,1000))

    BATCH = True

    # Generate initial conditions
    Nt = 1000
    # state = np.array([-10, 55, -80, 0.0, 0.0, 0.0])
    state = np.array([r0, rdot0]).flatten()

    # state = np.zeros(6)
    dt = 20 #seconds
    # ts = np.linspace(1,2000,Nt)

    # Generate random maneuvers and random time indices
    Ndv = 10
    dvd = np.random.normal(0.0, 0.000001, (Ndv,3))
    plt.title("$\Delta$V")
    plt.hist(np.linalg.norm(dvd,axis=1), bins=100)
    plt.show()
    dvi = np.random.choice(np.arange(Nt), size=Ndv, replace=False)
    print(dvi * dt, " [seconds]")
    
    dvs = np.zeros((Nt,6))
    for i,j in enumerate(dvi):
        dvs[j:,3:] += dvd[i]
    print(dvs.shape, dvs.dtype, dvs.sum(axis=0))

    # I think CW functions are actually only valid when not applying delta v, in other words, they are TIME INVARIANT!!!!!!!!

    if BATCH:
        states = CW_vectorized_batch(state, omeg, np.arange(0.0, Nt, dt))

    # else:
    #     for i in range(1,Nt):
    #         # Reason this wasn't working was because the CW function is time invariant and non-recursive like I was doing here
    #         states[i,:] = CW_vectorized_single(states[i-1]+dvs[i-1], omeg, dt, i)

    mask = [~np.any(~np.isfinite(r)) for r in states]
    # print(mask)
    # print(states.shape)
    states = states[mask]
    print(states)
    plt.hist(states, bins=100)
    plt.show()
    
    # states = states[np.isfinite(states)]
    # plt.hist(states, bins=100)
    # plt.show()
    # print(states.shape)

    ############################################################
    ############################################################
    # See if we can apply controls to maneuver in 'orbit' around target satellite
    plot_cw(states)

    ###########################################################
    ###########################################################

    ds = []
    xs = []
    ys = []
    zs = []

    d =0
    d_max = 50
    stindex = nframes
    for i in range(nframes):
        t = dt*i
        if d<d_max:
            r_vec, rdot_vec = CW2(r0, rdot0, omeg, t)
            # rdot_vec = CW2(r0, rdot0, omeg, t)[1]
            x = r_vec[0]
            y = r_vec[1]
            z = r_vec[2]
            xdot = rdot_vec[0]
            ydot = rdot_vec[1]
            zdot = rdot_vec[2]
            rdot = [xdot, ydot, zdot]
            d = np.linalg.norm(r_vec)
            ds.append(d)
            
            xs.append(x)
            ys.append(y)
            zs.append(z)
            
            v = np.linalg.norm(rdot_vec)
            stindex = i
        
    #second manouvre
    r0 = [x, y, z]
    deltav = [-0.03, 0, 0.02]
    for i in range(len(rdot)):
        rdot0[i] = rdot[i]+deltav[i]


    for i in range(nframes2):
        t = dt*i
        r_vec, rdot_vec = CW2(r0, rdot0, omeg, t)
        x = r_vec[0]
        y = r_vec[1]
        z = r_vec[2]
        xdot = rdot_vec[0]
        ydot = rdot_vec[1]
        zdot = rdot_vec[2]
        rdot = [xdot, ydot, zdot]
        d = np.sqrt(x**2+y**2+z**2)
        ds.append(d)
        
        xs.append(x)
        ys.append(y)
        zs.append(z)
        
        v = np.sqrt(xdot**2+ydot**2+zdot**2)

    #maonoeuvre 3

    r0 = [x, y, z]
    deltav = [0, -0.05, 0.03]
    for i in range(len(rdot)):
        rdot0[i] = rdot[i]+deltav[i]
    

    for i in range(nframes3):
        t = dt*i
        r_vec, rdot_vec = CW2(r0, rdot0, omeg, t)
        x = r_vec[0]
        y = r_vec[1]
        z = r_vec[2]
        xdot = rdot_vec[0]
        ydot = rdot_vec[1]
        zdot = rdot_vec[2]
        
        d = np.sqrt(x**2+y**2+z**2)
        ds.append(d)
        
        xs.append(x)
        ys.append(y)
        zs.append(z)
        
        v = np.sqrt(xdot**2+ydot**2+zdot**2)

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    zmin = min(zs)
    zmax = max(zs)
    rmax = max(ds)




    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d',xlim = (xmin, xmax), ylim = (ymin, ymax),zlim = (zmin, zmax))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')  


    frame = 0
    while frame < stindex:
        frame +=1
        ax.scatter(xs[frame], ys[frame], zs[frame], marker = '.', color = 'g', alpha = 0.5, s=1)
        
    while frame < nframes2+stindex:
        frame += 1
        ax.scatter(xs[frame], ys[frame], zs[frame], marker = '.', color = 'skyblue', s=1)
        
    while frame < nframes3+nframes2+stindex:
        frame += 1
        ax.scatter(xs[frame], ys[frame], zs[frame], marker = '.', color = 'y', s=1)

    plt.show()

    # Check equivalence of batch method and CW2 for-each method
    print("Batch and CW2 match? ", np.allclose(states[:,:3], np.c_[xs, ys, zs][:len(states)]))