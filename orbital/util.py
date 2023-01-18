from loadtle_example import *

import numpy as np

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential

from poliastro.util import time_range
from poliastro.ephem import Ephem
from poliastro.frames import Planes

import codecs

def load_CZML(fpath):
    return codecs.open(fpath).read()

def save_viewer(doc, fpath):
    with open(fpath, 'w') as f:
        f.write(str(doc).strip('\n'))

def load_tle_file(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    return [Satrec.twoline2rv(s,t) for s,t in zip(lines[1::2], lines[2::2])]

def rvt_to_ephem(rs, vs, times):
    ephems = []
    for i in range(len(sats)):

        cart_teme = CartesianRepresentation(
            rs[i] << u.km,
            xyz_axis=-1,
            differentials=CartesianDifferential(
                vs[i] << (u.km / u.s),
                xyz_axis=-1,
            )
        )

        cart_gcrs = (
            TEME(cart_teme, obstime=times)
            .transform_to(GCRS(obstime=times))
            .cartesian
        )

        ephems.append(Ephem(cart_gcrs, times, plane=Planes.EARTH_EQUATOR))

    return ephems

def gp_to_ephem(sats, times):
    errors, rs, vs = sats.sgp4(times.jd1, times.jd2)
    if not (errors == 0).all():
        warn(
            "Some objects could not be propagated, "
            "proceeding with the rest",
            stacklevel=2,
        )
        rs = rs[errors == 0]
        vs = vs[errors == 0]
        times = times[errors == 0]

    ephems = []
    for i in range(len(sats)):

        cart_teme = CartesianRepresentation(
            rs[i] << u.km,
            xyz_axis=-1,
            differentials=CartesianDifferential(
                vs[i] << (u.km / u.s),
                xyz_axis=-1,
            )
        )

        cart_gcrs = (
            TEME(cart_teme, obstime=times)
            .transform_to(GCRS(obstime=times))
            .cartesian
        )

        ephems.append(Ephem(cart_gcrs, times, plane=Planes.EARTH_EQUATOR))

    return ephems

def ephem_to_czml(ephems, times):
    from poliastro.czml.extract_czml import CZMLExtractor

    ext = CZMLExtractor(times[0], times[-1], len(times))

    for e in ephems:
        r, v = e.rv(times)
        cart_gcrs = CartesianRepresentation(
            r, 
            xyz_axis=-1, 
            differentials=CartesianDifferential(
                v, 
                xyz_axis=-1
                )
            )

        # Trajectory is more versatile, orbit is a little more rigid I think
        ext.add_trajectory(cart_gcrs, times)

    return str(ext.get_document())

def create_viewer(czml_str):
    """For now, this is a verbatim HTML string. In the future, we should probably use a templating library here"""

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <script src="https://cesium.com/downloads/cesiumjs/releases/1.98/Build/Cesium/Cesium.js"></script>
  <link href="https://cesium.com/downloads/cesiumjs/releases/1.98/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
</head>
<body>
  <div id="cesiumContainer"></div>
  <script>
    var viewer = new Cesium.Viewer("cesiumContainer", {{
        shouldAnimate: true,
    }});

    Cesium.CzmlDataSource.load(Object({czml_str})).then(function (ds) {{
        viewer.dataSources.add(ds);
        viewer.trackedEntity = ds.entities.getById(0);
    }});
  </script>
</body>
</html>
"""

def plot_2d_orbit(ephem, time):
    from poliastro.twobody import Orbit
    from poliastro.bodies import Earth
    orbit = Orbit(ephem, time)
    orbit.plot()

if __name__=="__main__":

    SAT_NAME = "ISS (Zarya)"

    import sys
    if len(sys.argv) > 1:
        SAT_NAME = str(sys.argv[1])

    # Load general pertubation data from online into satellite records
    sats = list(load_gp_from_celestrak(name="ISS (Zarya)"))
    print(sats)

    # Sample satellite records at various explicit times
    t0 = Time.now() - (1.0 << u.h)
    t1 = t0 + (1.0 << u.h)
    times = time_range(t0, end=t1, periods=100)
    # print(times)
    
    # err, pos, vel
    # errors, rs, vs = sat[0].sgp4_array(times.jd1, times.jd2)

    # Convert satellite records to ephemerides
    ephems = [ephem_from_gp(sat, times) for sat in sats]
    # print(ephems)

    # Write ephemerides to CZML representation
    czml_str = ephem_to_czml(ephems, times)
    # print(czml_str)

    # Write CZML to file, or write out a static HTML document with embedded CZML viewer
    viewer = create_viewer(czml_str)

    # Optionally, write out the viewer to HTML
    from pathlib import Path
    save_viewer(viewer, str(Path.cwd() / Path("../index.html")))