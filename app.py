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
import collections

# Initial page config

st.set_page_config(
     page_title='Estimate Programmer Compensation',
     layout="wide",
     initial_sidebar_state="expanded",
     page_icon="favicon.png"
)

def main():
    
    # Main code for model goes here
    st.header("BUS 458 Final Project Compensation Estimate")
    st.subheader("This app is intended to demonstrate our data analytics abilities in BUS 458. It runs a linear regression model based on the Kaggle Programmer Compensation Survey.")

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
    
    InputGender = "Man" + ';'
    
    InputCountry = st.selectbox('What country do you live in?', kaggleCon['Country'].unique()) + ';'
    
    InputStudent = str(int(st.toggle('Are you a student?'))) + ';'
    
    InputYearsProgramming = st.selectbox('For how many years have you been writing code and/or programming?', kaggleCon['Years.Programming'].unique()) + ';'
    
    InputIncorporateMachineLearning = st.selectbox('How does your company incorporate machine learning?', kaggleCon['Incorporate.Machine.Learning'].unique()) + ';'
    
    InputMLHubsRepositoriesUsed = st.selectbox('What Machine Learning Repository do you use?', kaggleCon['ML.Hubs...Repositories.Used'].unique()) + ';'
    
    InputHighestLevelofFormalEducation = st.selectbox('What is the highest level of formal education that you have attained or plan to attain within the next 2 years?', kaggleCon['Highest.Level.of.Formal.Education'].unique()) + ';'

    InputProgrammingLanguages = st.multiselect('What programming languages do you use on a regular basis? (Select all that apply)', ['Python', 'R', 'SQL', 'C', 'C#', 'C++', 'Java', 'Javascript', 'Bash', 'PHP', 'MATLAB', 'Julia', 'Go' ])

    NoProgramming = '0'
    if len(InputProgrammingLanguages) == 0:
        NoProgramming = '1'

    Go = '0;'
    if 'Go' in InputProgrammingLanguages:
        Go = '1;'

    Julia = '0;'
    if 'Julia' in InputProgrammingLanguages:
        Julia = '1;'

    mlab = '0;'
    if 'MATLAB' in InputProgrammingLanguages:
        mlab = '1;'

    php = '0;'
    if 'PHP' in InputProgrammingLanguages:
        php = '1;'
    
    basher = '0;'
    if 'Bash' in InputProgrammingLanguages:
        basher = '1;'

    jscript = '0;'
    if 'Javascript' in InputProgrammingLanguages:
        jscript = '1;'

    jva = '0;'
    if 'Java' in InputProgrammingLanguages:
        jva = '1;'
    
    cplus = '0;'
    if 'C++' in InputProgrammingLanguages:
        cplus = '1;'
    
    chash = '0;'
    if 'C#' in InputProgrammingLanguages:
        chash = '1;'
    
    c = '0;'
    if 'C' in InputProgrammingLanguages:
        c = '1;'
    
    sql = '0;'
    if 'SQL' in InputProgrammingLanguages:
        sql = '1;'
    
    r = '0;'
    if 'R' in InputProgrammingLanguages:
        r = '1;'
    
    python = '0;'
    if 'Python' in InputProgrammingLanguages:
        python = '1;'

    InputHelpfulCourses = st.multiselect('What products or platforms did you find to be most helpful when you first started studying data science? (Select all that apply)', ['University courses', 'Online courses (Coursera, EdX, etc)', 'Social media platforms (Reddit, Twitter, etc)', 'Video platforms (YouTube, Twitch, etc)', 'Kaggle (notebooks, competitions, etc)', 'None / I do not study data science'])

    university = '0;'
    if 'University courses' in InputHelpfulCourses:
        university = '1;'

    online = '0;'
    if 'Online courses (Coursera, EdX, etc)' in InputHelpfulCourses:
        online = '1;'
    
    social = '0;'
    if 'Social media platforms (Reddit, Twitter, etc)' in InputHelpfulCourses:
        social = '1;'
    
    video = '0;'
    if 'Video platforms (YouTube, Twitch, etc)' in InputHelpfulCourses:
        video = '1;'
    
    Helpfulkaggle = '0;'
    if 'Kaggle (notebooks, competitions, etc)' in InputHelpfulCourses:
        Helpfulkaggle = '1;'

    HelpfulNone = '0;'
    if 'Kaggle (notebooks, competitions, etc)' in InputHelpfulCourses:
        HelpfulNone = '1;'

    InputMediaSources = st.multiselect( 'Who/what are your favorite media sources that report on data science topics? (Select all that apply)', [
        'Twitter (data science influencers)',
        'Email newsletters (Data Elixir, O\'Reilly Data & AI, etc)',
        'Reddit (r/machinelearning, etc)',
        'Kaggle (notebooks, forums, etc)',
        'Course Forums (forums.fast.ai, Coursera forums, etc)',
        'YouTube (Kaggle YouTube, Cloud AI Adventures, etc)',
        'Podcasts (Chai Time Data Science, O’Reilly Data Show, etc)',
        'Blogs (Towards Data Science, Analytics Vidhya, etc)',
        'Journal Publications (peer-reviewed journals, conference proceedings, etc)',
        'Slack Communities (ods.ai, kagglenoobs, etc)',
        'None'
    ])

    MediaSourceMap = {
        'Twitter (data science influencers)': '0;',
        'Email newsletters (Data Elixir, O\'Reilly Data & AI, etc)':'0;',
        'Reddit (r/machinelearning, etc)':'0;',
        'Kaggle (notebooks, forums, etc)':'0;',
        'Course Forums (forums.fast.ai, Coursera forums, etc)':'0;',
        'YouTube (Kaggle YouTube, Cloud AI Adventures, etc)':'0;',
        'Podcasts (Chai Time Data Science, O’Reilly Data Show, etc)':'0;',
        'Blogs (Towards Data Science, Analytics Vidhya, etc)':'0;',
        'Journal Publications (peer-reviewed journals, conference proceedings, etc)':'0;',
        'Slack Communities (ods.ai, kagglenoobs, etc)':'0;',
        'None':'0;'
    }

    for key, value in MediaSourceMap.items():
        if key in InputMediaSources:
            MediaSourceMap[ key ] = '1;'

    StringData = StringIO('Age;Gender;Country;Student;Years.Programming;Incorporate.Machine.Learning;ML.Hubs...Repositories.Used;Highest.Level.of.Formal.Education;Helpful.University;' 
                  + 'Helpful.Online.Courses;Helpful.Social.Media;Helpful.Video.Platform;Helpful.Kaggle;Helpful.None;Media.on.Social.Twitter;Media.on.Social.Email.Newsletters;'
                  + 'Media.on.Reddit;Media.on.Kaggle;Media.on.Course.Forums;Media.on.Youtube;Media.on.Podcasts;Media.on.Blogs;Media.on.Journal.Publications;Media.on.Slack.Communities;'
                  + 'No.Media.Sources;Data.Science.on.Coursera;Data.Science.on.edX;Data.Science.on.Kaggle.Learn.Courses;Data.Science.on.DataCamp;Data.Science.on.Fast.ai;Data.Science.on.Udacity;Data.Science.on.Udemy;'
                  + 'Data.Science.on.LinkedIn.Learning;Cloud.certification.programs;Data.Science.University.Courses;No.Data.Science.Courses;Python;R;SQL;C;C.;C..;Java;Javascript;Bash;PHP;MATLAB;Julia;Go;No.Programming.Languages\n'
                  + InputAge + InputGender + InputCountry + InputStudent + InputYearsProgramming + InputIncorporateMachineLearning +
                          InputMLHubsRepositoriesUsed + InputHighestLevelofFormalEducation + university + online + social + video + Helpfulkaggle + HelpfulNone + 
                          MediaSourceMap['Twitter (data science influencers)'] + MediaSourceMap['Email newsletters (Data Elixir, O\'Reilly Data & AI, etc)'] +  MediaSourceMap['Reddit (r/machinelearning, etc)']
                           + MediaSourceMap['Kaggle (notebooks, forums, etc)'] + MediaSourceMap['Course Forums (forums.fast.ai, Coursera forums, etc)'] 
                           + MediaSourceMap['YouTube (Kaggle YouTube, Cloud AI Adventures, etc)']  + MediaSourceMap['Podcasts (Chai Time Data Science, O’Reilly Data Show, etc)'] + MediaSourceMap['Blogs (Towards Data Science, Analytics Vidhya, etc)']
                           + MediaSourceMap['Journal Publications (peer-reviewed journals, conference proceedings, etc)'] + MediaSourceMap['Slack Communities (ods.ai, kagglenoobs, etc)'] + MediaSourceMap['None']
                           + '0;' + '0;' + '0;' + '0;' + '0;'  + '0;' + '0;' + '0;' + '0;' + '0;'
                           + '0;' + python + r + sql + c  + chash + cplus + jva + jscript + basher + php + mlab + Julia + Go + NoProgramming + '\n' )
    
    inputdf = pd.read_csv( StringData, sep=";")

    inputdf = pd.concat( [kaggleCon.drop(columns='Compensation'), inputdf] )
    
    input_encoded = pd.get_dummies(inputdf, columns=cols_to_convert)
    
    prediction = lm.predict( input_encoded )

    ind_prediction = prediction[len(prediction) - 1]
    
    st.metric(label="Your Predicted Compensation (USD)", value=ind_prediction)
    
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
