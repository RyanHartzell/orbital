import streamlit as st
import streamlit.components.v1 as components
import codecs
from pathlib import Path

def html_viewer(viewer, width=700, height=500):
    doc = codecs.open(viewer, 'r').read()
    components.html(doc, width=width, height=height, scrolling=False)

if __name__=="__main__":
    st.title("Basic CZML Viewer")

    html_viewer(str(Path.cwd() / Path("test.html")))