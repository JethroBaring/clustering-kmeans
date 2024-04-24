import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# Set page configuration
st.set_page_config(page_title="Wine Clustering")

# Load the KMeans model
kmeans = pickle.load(open('wine_clustering.pkl', 'rb'))

# Streamlit app title
st.title("Wine Clustering App")

# Function to predict cluster for new data
def predict_cluster(input_array, kmeans_model):
    # Predict cluster
    cluster = kmeans_model.predict(input_array.reshape(1, -1))
    return cluster[0]

labels = [
    "Alcohol", "Malic Acid", "Ash", "Ash Alcanity", "Magnesium",
    "Total Phenols", "Flavanoids", "Nonflavanoid Phenols",
    "Proanthocyanins", "Color Intensity", "Hue", "OD280", "Proline"
]

# Divide labels into 4 columns
col1, col2, col3, col4 = st.columns(4)

# Initialize input values dictionary
input_values = {}

# Input fields for each feature
for i, label in enumerate(labels):
    if i < 4:
        col = col1
    elif i < 8:
        col = col2
    elif i < 12:
        col = col3
    else:
        col = col4
    input_values[label] = col.number_input(label, step=0.01)

# Convert input values to numpy array
input_array = np.array(list(input_values.values()))

# Perform PCA on existing data
data = pd.read_csv("wine-clustering.csv")
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

# Create trace for existing data points with cluster colors
traces = []
colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)']  # Define distinct colors for clusters
for i in range(kmeans.n_clusters):
    trace = go.Scatter(
        x=reduced_data[kmeans.labels_ == i, 0],
        y=reduced_data[kmeans.labels_ == i, 1],
        mode='markers',
        marker=dict(color=colors[i], size=8),
        name=f'Cluster {i}',
        showlegend=True
    )
    traces.append(trace)

# Initialize the plot with existing data
fig = go.Figure(data=traces)

# Update layout
fig.update_layout(
    title='Wine Clustering Plot',
    xaxis_title='Principal Component 1',
    yaxis_title='Principal Component 2',
    hovermode='closest',
    plot_bgcolor='rgba(255,255,255,1)'  # Set background color
)

cluster = None
hasInput = True

# Function to update the plot with new data
def update_plot(input_values, kmeans_model, pca_model, fig):
    global cluster, hasInput
    # Convert input values to numpy array
    input_array = np.array(list(input_values.values())).reshape(1, -1)
    # Predict cluster for new data
    cluster = predict_cluster(input_array, kmeans_model)
    hasInput = True
    # Perform PCA on new data point
    reduced_new_data = pca_model.transform(input_array)
    # Add trace for new data point
    fig.add_trace(
        go.Scatter(
            x=reduced_new_data[:, 0],
            y=reduced_new_data[:, 1],
            mode='markers',
            marker=dict(color='red', size=12, symbol='star'),
            name='New Data',
            showlegend=True
        )
    )

# Button to trigger prediction and plot update
if st.button("Predict Cluster"):
    update_plot(input_values, kmeans, pca, fig)

# Display the plot
st.plotly_chart(fig)

# Display predicted cluster if available
if cluster is not None:
    st.write("Predicted Cluster:", cluster)

# Display message if no input is provided
if not hasInput:
    st.write("Please enter values for all features.")
