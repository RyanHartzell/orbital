# orbital
Simple hacky project for viewing CesiumJS+HTML document in a Streamlit app

Find it deployed on Streamlit Community Cloud here:
[RyanHartzell - Orbital - CZML Viewer](https://ryanhartzell-orbital-czmlviewer.streamlit.app/)

Project includes:

- Basic Streamlit template app (might register as a streamlit plugin at some point or something...
- Formatting functions for injecting CZML json object into the HTML (either streaming or static)
- Basic helpers for loading TLE data or explicitly defined orbital element files using Python

Goal is for this to be a "bare minimum" interactive CZML formatter and viewer using a Python backend. Use cases (rather than just using CesiumJS directly in the browser) are mainly that this enables web-based interactive viewing of CZML scenes that are backed by potentially complex, existing server-side Python workflows.
