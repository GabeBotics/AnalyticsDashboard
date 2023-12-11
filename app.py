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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from io import StringIO

# Initial page config

st.set_page_config(
     page_title='Estimate Programmer Compensation',
     layout="wide",
     initial_sidebar_state="expanded",
     page_icon="favicon.png"
)

def main():
    
    # Main code for model goes here

    # url='https://drive.google.com/file/d/1sNJ7KwzX93RGZQQhXfm1R6bLcNRUiBcP/view?usp=sharing'
    # url='https://drive.google.com/uc?id=' + url.split('/')[-2]

    # Read the CSV file
    kaggleCon = pd.read_csv( 'PythonKaggleFile.csv', index_col=0 )
    
    # Convert specified columns to factors
    cols_to_convert = ['Age', 'Gender', 'Country', 'Years.Programming', 'Incorporate.Machine.Learning', 'ML.Hubs...Repositories.Used', 'Highest.Level.of.Formal.Education']
    kaggleCon[cols_to_convert] = kaggleCon[cols_to_convert].apply(lambda x: x.astype('category'))
    
    df_encoded = pd.get_dummies(kaggleCon, columns=cols_to_convert)
    
    # Define the dependent variable and independent variables for the linear model
    y = df_encoded['Compensation']
    X = df_encoded.drop(columns=['Compensation'])
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=29)
    
    lm = LinearRegression().fit( X_train, y_train )
    
    InputAge = st.selectbox('How old are you?', ["18-21","22-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-69","70+"]) + ';'
    
    InputGender = st.radio(
    "What is your gender?", ["Man", "Woman", "Prefer not to say", "Nonbinary", "Prefer to self-describe"]) + ';'
    
    InputCountry = 'United States of America' + ';'
    
    InputStudent = str(1) + ';'
    
    InputYearsProgramming = '3-5 years' + ';'
    
    InputIncorporateMachineLearning = 'No Answer' + ';'
    
    InputMLHubsRepositoriesUsed = 'No Answer' + ';'
    
    InputHighestLevelofFormalEducation = 'Some college/university study without earning a bachelor’s degree' + ';'
    
    StringData = StringIO('Age;Gender;Country;Student;Years.Programming;Incorporate.Machine.Learning;ML.Hubs...Repositories.Used;Highest.Level.of.Formal.Education;Helpful.University;' 
                  + 'Helpful.Online.Courses;Helpful.Social.Media;Helpful.Video.Platform;Helpful.Kaggle;Helpful.None;Media.on.Social.Twitter;Media.on.Social.Email.Newsletters;'
                  + 'Media.on.Reddit;Media.on.Kaggle;Media.on.Course.Forums;Media.on.Youtube;Media.on.Podcasts;Media.on.Blogs;Media.on.Journal.Publications;Media.on.Slack.Communities;'
                  + 'No.Media.Sources;Data.Science.on.Coursera;Data.Science.on.edX;Data.Science.on.Kaggle.Learn.Courses;Data.Science.on.DataCamp;Data.Science.on.Fast.ai;Data.Science.on.Udacity;Data.Science.on.Udemy;'
                  + 'Data.Science.on.LinkedIn.Learning;Cloud.certification.programs;Data.Science.University.Courses;No.Data.Science.Courses;Python;R;SQL;C;C.;C..;Java;Javascript;Bash;PHP;MATLAB;Julia;Go;No.Programming.Languages\n'
                  + InputAge + InputGender + InputCountry + InputStudent + InputYearsProgramming + InputIncorporateMachineLearning +
                          InputMLHubsRepositoriesUsed + InputHighestLevelofFormalEducation + '0;' + '0;' + '0;' + '0;' + '0;' + '0;' + '0;' + '0;' + '0;'
                           + '0;' + '0;' + '0;'  + '0;' + '0;' + '0;' + '0;' + '0;' + '0;' + '0;' + '0;' + '0;' + '0;'  + '0;' + '0;' + '0;' + '0;' + '0;'
                           + '0;' + '0;' + '0;' + '0;' + '0;'  + '0;' + '0;' + '0;' + '0;' + '0;' + '0;' + '0;' + '0;' + '0;' + '0\n' )
    
    inputdf = pd.read_csv( StringData, sep=";")

    inputdf = pd.concat( [kaggleCon.drop(columns='Compensation'), inputdf] )
    
    input_encoded = pd.get_dummies(inputdf, columns=cols_to_convert)
    
    prediction = lm.predict( input_encoded )

    ind_prediction = prediction[len(prediction) - 1]
    
    st.metric(label="Your Predicted Compensation", value=ind_prediction)
    
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
    st.sidebar.header('Predict Programmer Compensation')
    st.sidebar.subheader('BUS 458 Final Project')
    return None

##########################
# Main body of cheat sheet
##########################

def cs_body():
    
    return None

# Run main()

if __name__ == '__main__':
    main()
