"""
Streamlit Cheat Sheet

App to summarise streamlit docs v1.25.0

There is also an accompanying png and pdf version

https://github.com/daniellewisDL/streamlit-cheat-sheet

v1.25.0
20 August 2023

Author:
    @daniellewisDL : https://github.com/daniellewisDL

Contributors:
    @arnaudmiribel : https://github.com/arnaudmiribel
    @akrolsmir : https://github.com/akrolsmir
    @nathancarter : https://github.com/nathancarter

"""

import streamlit as st
from pathlib import Path
import base64

# Initial page config

st.set_page_config(
     page_title='BUS 458 Final Dashboard',
     layout="wide",
     initial_sidebar_state="expanded",
     page_icon="favicon.png"
)

def main():
    cs_sidebar()
    cs_body()

    return None

# Thanks to streamlitopedia for the following code snippet

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

# sidebar

def cs_sidebar():

    st.sidebar.markdown('''[<img src='data:image/png;base64,{}' class='img-fluid' width=148 height=148>](https://github.com/GabeBotics/AnalyticsDashboard)'''.format(img_to_bytes("logomark_website.png")), unsafe_allow_html=True)
    st.sidebar.header('BUS 458 Final Dashboard')
    return None

##########################
# Main body of cheat sheet
##########################

def cs_body():
    
    return None

# Run main()

if __name__ == '__main__':
    main()
