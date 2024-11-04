import streamlit as st
import streamlit.components.v1 as components
import codecs
from pathlib import Path
from loadtle_example import *
from util import *
from pathlib import Path
import httpx
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

custom_style = {'axes.labelcolor': 'lightblue',
                'xtick.color': 'lightblue',
                'ytick.color': 'lightblue'}
sns.set_style("darkgrid", rc=custom_style)

import altair as alt

def html_viewer(viewer, width=700, height=500):
    doc = codecs.open(viewer, 'r').read()
    components.html(doc, width=width, height=height, scrolling=False)

def html_viewer_live(viewer, **kwargs):
    components.html(viewer, **kwargs)

@st.cache_resource
def get_all_active_satellites():
    r = httpx.get("https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=xml")    
    xml = ET.fromstring(r.text)

    satdict = {k.text:Satrec() for k in xml.findall('.//OBJECT_NAME')}
    active_sat_xml = list(omm.parse_xml(io.StringIO(r.text))) # all segments in order (I hope...)    
    for k,seg in zip(satdict.keys(), active_sat_xml):
        omm.initialize(satdict[k], seg)

    # Make a df of GP data
    df = make_gp_df(xml)
    return satdict, df

def make_gp_df(xml):
    d = [[{x.tag:x.text for x in i} for i in s] for s in xml.findall('.//data')]
    d = [{**x, **y} for x,y in d]
    cols = list(d[0].keys())
    df = pd.DataFrame(data=[list(i.values()) for i in d], columns=cols)

    df['EPOCH'] = pd.to_datetime(df['EPOCH'])
    for c in ['MEAN_MOTION','ECCENTRICITY','INCLINATION','RA_OF_ASC_NODE','ARG_OF_PERICENTER','MEAN_ANOMALY','EPHEMERIS_TYPE','BSTAR','MEAN_MOTION_DOT','MEAN_MOTION_DDOT']:
        df[c] = pd.to_numeric(df[c])
    return df

def scatter_plot(df, x='MEAN_MOTION', y='MEAN_ANOMALY', color='ARG_OF_PERICENTER', tooltips=['NORAD_CAT_ID']):
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x=x,
        y=y,
        color=color,
        tooltip=tooltips
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

def heatmap(df):
    fig, ax = plt.subplots(facecolor='black')
    ax.set_title("Pearson Cross-Correlation Heatmap", color='lightblue')
    sns.heatmap(df.corr(), ax=ax, vmin=-1., vmax=1.)
    st.pyplot(fig)

if __name__=="__main__":
    # Load (and cache) our list of satellite names for multiselect
    satdict, df = get_all_active_satellites()

    st.title("Basic CZML Viewer Using Python for Ephem/Trajectory Generation")
    st.write("This viewer allows you to choose a particular (most recent) TLE from Celestrak by name and display them in an embedded CZML document.")
    
    # I want to be able to switch between pages so need multipage app navigation on the sidebar
    with st.sidebar:
        st.write("Lorem ipsum yada yada yada")

    # st.logo('Mines-Logo-triangle-blue.png', size='large')

    tabs = st.tabs(["CesiumJS Orbit Visualizer", "Data Exploration: Relative RSO Density", "Data Exploration: Scatter Plot", "Data Exploration: Correlation Heatmap"])

    with tabs[0]:
        # Page 1 - Loading and viewing satellites (need to add a checkbox for visualizing all of them?)
        # Should also add the ability to select a particular time window

        satname_cols = st.columns([5,1])

        with satname_cols[0]:
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
        with st.container():    
            html_viewer_live(viewer, height=500, scrolling=True)
            st.write("*If the viewer disappears or shrinks, reload the page or interact with the select box above.*")

    with tabs[1]:
        # Page 2 - Density stuff
        # Select a particular host satellite (AzEl reference frame should just be RIC I think...)
        # Show density plots in either rectilinear full sky, targeted image space view of targets, or a polar plot view at each time step
        # Should be "scrollable" or animated via a slider or similar functionality to play the animation of density over time
        # Should allow histogram or marginal density plots and vmin vmax settings for contrast (as well as logarithmic scaling)
        st.empty()

    with tabs[2]:
        # Page 3 - Allows data exploration (optionally tSNE?)
        # Will show the full altair plot on this page and allow filtering the dataframe of ALL satellite GP info for the time range
        # This is just GP data!!!! No derivative values plz
        st.write("Data Exploration Scatter Plot")
        with st.container():
            cols = st.columns(3)
            xdata = cols[0].selectbox('X Data', df.columns, 1)
            ydata = cols[1].selectbox('Y Data', df.columns, 2)
            colordata = cols[2].selectbox('Color', df.columns, 5)

            # Plot scatter plot with up-to-date choices of columns
            scatter_plot(df, x=xdata, y=ydata, color=colordata)
    
    # Page 4: Show correlation heatmap
    with tabs[3]:
        # Heatmap
        heatmap(df.select_dtypes('number'))
        # Viz of dataframe
        st.dataframe(df)