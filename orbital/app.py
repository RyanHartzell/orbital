import streamlit as st
import streamlit.components.v1 as components
import codecs
from pathlib import Path
from loadtle_example import *
from util import *
from pathlib import Path
import httpx
import xml.etree.ElementTree as ET

def html_viewer(viewer, width=700, height=500):
    doc = codecs.open(viewer, 'r').read()
    components.html(doc, width=width, height=height, scrolling=False)

def html_viewer_live(viewer, **kwargs):
    components.html(viewer, **kwargs)

@st.cache_resource
def get_all_active_satellites():
    r = httpx.get("https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=xml")
    satdict = {k:Satrec() for k in [n.text for n in ET.fromstring(r.text).findall('.//OBJECT_NAME')]}
    active_sat_xml = list(omm.parse_xml(io.StringIO(r.text))) # all segments in order (I hope...)
    
    for k,seg in zip(satdict.keys(), active_sat_xml):
        omm.initialize(satdict[k], seg)

    return satdict

# def gen_sats_by_name(names, satdict):    
#     # Need to search each xml segment separately? I think?
#     for name in names:
#         yield satdict[name]

# @st.cache_data
# def get_active_sat_names():
#     r = httpx.get("https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json")
#     return [s["OBJECT_NAME"] for s in r.json()]

if __name__=="__main__":
    # Load (and cache) our list of satellite names for multiselect
    satdict = get_all_active_satellites()

    st.title("Basic CZML Viewer Using Python for Ephem/Trajectory Generation")
    st.write("This viewer allows you to choose a particular (most recent) TLE from Celestrak by name and display them in an embedded CZML document.")
    
    options = st.multiselect(
        "Please select a few active satellites by name: ",
        satdict.keys(),
        ["ISS (ZARYA)"],
    )

    # Load general pertubation data from online into satellite records
    sats = [satdict[satname] for satname in options]

    # Sample satellite records at various explicit times
    t0 = Time.now() - (1.0 << u.h)
    t1 = t0 + (1.0 << u.h)
    times = time_range(t0, end=t1, periods=100)

    # Convert satellite records to ephemerides
    ephems = [ephem_from_gp(sat, times) for sat in sats]

    # Write ephemerides to CZML representation
    czml_str = ephem_to_czml(ephems, times)

    # Write CZML to file, or write out a static HTML document with embedded CZML viewer
    viewer = create_viewer(czml_str)

    # Optionally, write out the viewer to HTML
    # save_viewer(viewer, str(Path.cwd() / Path("test.html")))

    # html_viewer(str(Path.cwd() / Path("test.html")))
    html_viewer_live(viewer, width=700, height=500, scrolling=False)
