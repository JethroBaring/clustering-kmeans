import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# Load the KMeans model
kmeans = pickle.load(open('wine_clustering.pkl', 'rb'))

# Streamlit app title
st.title("Wine Clustering App")

# Function to predict cluster for new data
def predict_cluster(input_text):
    # Convert input text to a numpy array of float values
    input_array = np.array(input_text.split(','), dtype=np.float64)
    input_array = input_array.reshape(1, -1)  # Reshape to match model input

    # Predict cluster
    cluster = kmeans.predict(input_array)

    return cluster[0]

# Input field for comma-separated values
input_text = st.text_input("Enter comma-separated values")

# Button to trigger prediction
if st.button("Predict Cluster"):
    if input_text:
        cluster = predict_cluster(input_text)
        st.write("Predicted cluster:", cluster)
    else:
        st.write("Please enter values separated by commas.")