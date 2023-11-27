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

# Initial page config

st.set_page_config(
     page_title='BUS 458 Final Dashboard',
     layout="wide",
     initial_sidebar_state="expanded",
     page_icon="favicon.png"
)

def main():
    # Main code for model goes here

    url='https://drive.google.com/file/d/13FtnUNQTJvqrB-jJBCDbwprSuehISIk8/view?usp=sharing'
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]

    df = pd.read_csv(url, index_col=0)

    df.describe().loc[['mean', 'std']]

    scaler = StandardScaler()

    scaler.fit(df)

    x = scaler.transform(df)

    # Specify the number of clusters (you can change this as needed)
    num_clusters = 3
    
    # Extract the features for clustering
    features = df[['Rape', 'Assault']]
    
    # Initialize KMeans model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    
    # Fit the model and predict clusters
    df['cluster'] = kmeans.fit_predict(features)
    
    # Scatter plot for each cluster with a different color
    plt.figure(figsize=(8, 6))
    
    for cluster in range(num_clusters):
        cluster_data = df[df['cluster'] == cluster]
        plt.scatter(cluster_data['Rape'], cluster_data['Assault'], label=f'Cluster {cluster}')
    
    # Plot the centroids (optional)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, color='red', label='Centroids')
    
    plt.title('K-means Clustering')
    plt.xlabel('Rape')
    plt.ylabel('Assault')
    plt.legend()
    plt.show()
    
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
