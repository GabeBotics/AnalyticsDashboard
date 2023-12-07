"""
Analytics Dashboard for BUS 458 Final Project
Author:
    @GabeBotics : https://github.com/GabeBotics
"""

import streamlit as st
from pathlib import Path
import base64
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
 from rpy2 import robjects

# Initial page config

st.set_page_config(
     page_title='BUS 458 Final Dashboard',
     layout="wide",
     initial_sidebar_state="expanded",
     page_icon="favicon.png"
)

def main():
    # Main code for model goes here

    url='https://drive.google.com/file/d/1VJnyvUB2MwKKINfvpoYLHYeoHO7FznVp/view?usp=sharing'
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]

    df = pd.read_csv(url, index_col=0)

    robjects.r('''
         # create a function `f`
...         f <- function(r, verbose=FALSE) {
...             if (verbose) {
...                 cat("I am calling f().\n")
...             }
...             2 * pi * r
...         }
...         # call the function `f` with argument value 3
...         f(3)
...         ''')

    # Show the pyplot using streamlit's pyplot function
    st.pyplot( plt )
    
    # -----------------------------------
    
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
