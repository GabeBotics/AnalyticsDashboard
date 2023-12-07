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
        
        kaggleCon <- read.csv("kaggleContinuous.csv")

        kaggleCon <- kaggleCon %>%
            mutate(across(c(2:4, 6:14), as.factor))

        kaggleCon1 <- kaggleCon %>% 
            select(-c(Published.Academic.Research.Papers, How.many.individuals.are.responsible, Company.Size,Years.Used.Machine.Learning, Similar.Title, Industry.of.Work))

        lm <- lm(Compensation~ Age + Gender + Country + Years.Programming + Incorporate.Machine.Learning + ML.Hubs...Repositories.Used + Media.on.Reddit + Media.on.Course.Forums + Media.on.Podcasts + Media.on.Journal.Publications + Data.Science.on.Fast.ai + Data.Science.on.Udacity + Data.Science.University.Courses +Python+R+SQL+C+`C.`+`C..` + Java + Javascript + Bash + PHP + MATLAB + Julia + Go + No.Programming.Languages, data = kaggleCon1)

        summary(lm)

        vif(lm)

         ''')

    # Show the pyplot using streamlit's pyplot function
    # st.pyplot( plt )
    
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
