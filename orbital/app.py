import streamlit as st
import streamlit.components.v1 as components
import codecs
from pathlib import Path
from loadtle_example import *
from util import *
from density import *
from pathlib import Path
import httpx
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from datetime import timedelta
from access import in_major_keep_out_zones, not_sunlit, out_of_range
from skyfield.api import load
from access import MAX_LOS_RANGE, MIN_LOS_RANGE, MOON_KOZ, SUN_KOZ

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

    satdict = OrderedDict()
    for k in xml.findall('.//OBJECT_NAME'):
        satdict[k.text] = Satrec()

    active_sat_xml = list(omm.parse_xml(io.StringIO(r.text))) # all segments in order (I hope...)    
    for k,seg in zip(satdict.keys(), active_sat_xml):
        omm.initialize(satdict[k], seg)

    # Make a df of GP data
    df = make_gp_df(xml)
    return satdict, df

@st.cache_resource
def load_all_satellites():
    return load_satellites()

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

### ACCESS OPP WRAPPERS FOR CACHING
# Runs into cacheing issues due to use of non-hashable types. Could also supply hash funcs for the time objects and satellite objects instead..... that way they'd hash well
# in_major_keep_out_zones = st.cache_data(in_major_keep_out_zones)
# out_of_range = st.cache_data(out_of_range)
# not_sunlit = st.cache_data(not_sunlit)

if __name__=="__main__":
    # Load (and cache) our list of satellite names for multiselect
    satdict, df = get_all_active_satellites()
    # print(satdict, df)
    allsats = load_all_satellites()
    # print(allsats)

    # Edit for CSCI 598
    st.title("Density-Based Tasking of a Distributed Space-Telescope Network for Space Situational Awareness :sparkle: :telescope: :satellite:")
    st.header("Ryan Hartzell, CSCI 598, Colorado School of Mines")

    # st.title("Basic CZML Viewer Using Python for Ephem/Trajectory Generation")
    # st.write("This viewer allows you to choose a particular (most recent) TLE from Celestrak by name and display them in an embedded CZML document.")
    
    # # I want to be able to switch between pages so need multipage app navigation on the sidebar
    # with st.sidebar:
    #     st.write("Lorem ipsum yada yada yada")

    # st.logo('Mines-Logo-triangle-blue.png', size='large')

    tabs = st.tabs(["CesiumJS Orbit Visualizer", "Relative RSO Density", "Access Opportunity Generator", "DataEx: Scatter Plot", "DataEx: Correlation Heatmap"])

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
        t0 = Time.now() # - (1.0 << u.h)
        t1 = t0 + (1.0 << u.h)
        times = time_range(t0, end=t1, periods=60)

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
        
        # density_columns = st.columns([2,1,1,1])
        density_cols = st.columns([2,1])
        hostname = density_cols[0].selectbox("Select a host satellite for your space telescope:", satdict.keys())
        # date = st.date_input("Pick a date on which to calculate 24 hours of target density:")
        if density_cols[1].button("Run density pipeline"):

            st.markdown("*Note: The density calculator may take a while...")
            # host = density_columns[0].selectbox("Select a host satellite for your space telescope:", allsats)
            # density_start = density_columns[1].date_picker("Start [UTC]")
            # density_stop = density_columns[2].date_picker("Stop [UTC]")
            # density_step = density_columns[3].number_input("Step [minutes]")

            prog = st.progress(0.0, "Running density pipeline...")

            # Select a host and make targets a view of the rest of the stuff in that list of satellites
            # host = allsats[list(satdict.keys()).index(hostname)]
            # targets = [s for s in allsats if host != s]

            host = (targets := allsats.copy()).pop(list(satdict.keys()).index(hostname))

            # Get times
            ts = load.timescale()
            # times = ts.utc(date[0], date[1], date[2], 0, range(0,3*60+1,5)) # hourly, currently this is default
            t0 = ts.now()

            # Should add selectors for time window and step
            times = ts.utc(t0.utc_datetime() + np.asarray([timedelta(minutes=x) for x in range(0,60+1,5)])) # 5 minute steps over 1 hour

            # Computing default access opportunities
            prog.progress(.2, "Computing default target access from host...")

            sunlit_access = not_sunlit(times, targets)
            range_access = out_of_range(times, host, targets)
            koz_access = in_major_keep_out_zones(times, host, targets)

            access = ~sunlit_access * ~range_access * ~koz_access

            prog.progress(.4, "Calculating apparent RA/Dec of targets...")

            # Calculate apparent ra, dec, ranges relative to host state at each time t
            obs = reformat_radecrange(calculate_apparent_radecrange(host, np.asarray(targets), times, access), ragged=True)

            # Check for nans and filter out any row with at least one nan (and report in dictionary? maybe??)
            # print("SANITY CHECK: OBS SHAPE: ", obs.shape)
            # obs = obs[np.sum(np.isnan(obs), (1,2)).astype(bool),:,:]
            # print("SANITY CHECK: OBS SHAPE: ", obs.shape)

            # if np.any(np.isnan(obs)):
            #     prog.progress(1.0, "Abort.")
            #     st.write("% NaNs in RA/DEC/RANGE = ", np.sum(np.isnan(obs))/np.size(obs)*100.)
            #     st.write("# NaNs in RA/DEC/RANGE = ", np.sum(np.isnan(obs)))
            #     st.write("Aborting due to NaN values in observation table... :warning:")
            # else:

            try:

                prog.progress(.5, "Generating BallTree representations...")

                # Build all ball trees
                bts = [construct_ball_tree(obs[0][i], obs[1][i]) for i in range(len(times))]

                prog.progress(.6, "Building density maps...")

                # Build all density maps
                kde_maps = np.dstack([construct_kde_map(bt) for bt in bts])

                prog.progress(.95, "Almost done! ...")

                # Make animation of density maps!
                doc = animate_heatmaps(kde_maps, times, False)
                components.html(doc, height=600, scrolling=True)

                prog.progress(1.0, "Done! :sparkles:")

                dl = st.fragment(st.download_button('Download access-aware density animation!', data=doc, file_name='density.html', mime='text/html'))

            except Exception as e:
                st.write('There are likely nans in the observation table. Aborting... :poop:')
                raise e

            if dl:
                st.write('To display your downloaded animation, open it in your favorite browser!')

    with tabs[2]:
        st.write("Access Opportunities over Time")

        hostname = st.selectbox("Select a host satellite for your space telescope:", satdict.keys(), key='aoselect')
        host = (targets := allsats.copy()).pop(list(satdict.keys()).index(hostname))

        aocols = st.columns(3)
        aokoz = aocols[0].toggle("Activate Line-of-Sight/Keep-Out-Zone Constraint?", True)
        aorng = aocols[1].toggle("Activate Range Constraint?", True)
        aolit = aocols[2].toggle("Activate Direct Sun-Lighting Constraint?", True)

        # Set up constraint inputs in separate column structure, active by above bools
        aonumcols = st.columns(5)
        maxrng = aonumcols[1].number_input('Enter a maximum range [km]:', min_value=0.0, value=MAX_LOS_RANGE, disabled=not aorng, key='aomaxrng_num')
        minrng = aonumcols[0].number_input('Enter a minimum range [km]:', min_value=0.0, max_value=MAX_LOS_RANGE, value=MIN_LOS_RANGE, disabled=not aorng, key='aominrng_num')
        sunkoz = np.deg2rad(aonumcols[2].number_input('Enter Sun Keep Out Zone [deg]:', min_value=0.0, max_value=90., value=np.rad2deg(SUN_KOZ), disabled=not aokoz, key='aosunkoz_num'))
        moonkoz = np.deg2rad(aonumcols[3].number_input('Enter Moon Keep Out Zone [deg]:', min_value=0.0, max_value=90., value=np.rad2deg(MOON_KOZ), disabled=not aokoz, key='aomoonkoz_num'))
        earthkoz = aonumcols[4].number_input('Enter an altitude Earth KOZ [km]:', min_value=0.0, value=90., disabled=not aokoz)

        # Get times
        from datetime import timedelta
        ts = load.timescale()
        t0 = ts.now()
        times = ts.utc(t0.utc_datetime() + np.asarray([timedelta(minutes=x) for x in range(0, 361)])) # 360 minute (6 hour) timeframe

        access = np.ones((len(targets), len(times)))

        progao = st.progress(0.0, "Computing access opportunities...")
        amount_done = 0.0
        amount_done_inc = 1 / sum([aolit, aorng, aokoz])

        if aolit:
            sunlit_access = not_sunlit(times, targets)
            access = access * ~sunlit_access
            amount_done += amount_done_inc
            progao.progress(amount_done, "Computed direct sunlight access constraint.")

        if aorng:
            range_access = out_of_range(times, host, targets, min_r=minrng, max_r=maxrng)
            access = access * ~range_access
            amount_done += amount_done_inc
            progao.progress(amount_done, "Computed LOS range access constraint.")

        if aokoz:
            koz_access = in_major_keep_out_zones(times, host, targets, moon_koz_deg=moonkoz, sun_koz_deg=sunkoz, earth_alt_koz_pad_km=earthkoz)
            access = access * ~koz_access
            amount_done += amount_done_inc
            progao.progress(amount_done, "Computed KOZ access constraints.")

        # Construct overall access mask (should be SATNUM x TIMESTEP)
        # access = ~sunlit_access * ~range_access * ~koz_access # We can multiply these since any zero value should cause a switch to False
        # print(f"Total % access across timesteps = {np.sum(access)/access.size * 100.}")

        progao.progress(1.0, "Done! :sparkles:")

        # Plot access over time as total satellites available for observation at each timestep
        fig = plt.figure()
        plt.tick_params(axis='both', color='k', labelcolor='k')
        plt.plot(times.utc_datetime(), access.sum(0), label='Total Access')
        if aolit:
            plt.plot(times.utc_datetime(), (~sunlit_access).sum(0), '--', label='Sunlit Access')
        if aorng:    
            plt.plot(times.utc_datetime(), (~range_access).sum(0), '--', label='Range Access')
        if aokoz:
            plt.plot(times.utc_datetime(), (~koz_access).sum(0), '--', label='KOZ Access')
        
        plt.title(f"Valid Access Opportunities T+6.00 [hr]\n{host}", color='k')
        plt.xlabel("Datetime", color='k')
        plt.ylabel("# of Observable Targets", color='k')
        plt.legend(loc='upper right')
        st.pyplot(fig)

    with tabs[3]:
        # Page 4 - Allows data exploration (optionally tSNE?)
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
    
    # Page 5: Show correlation heatmap
    with tabs[4]:
        # Heatmap
        heatmap(df.select_dtypes('number'))
        # Viz of dataframe
        st.dataframe(df)