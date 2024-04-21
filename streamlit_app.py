import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
st.set_page_config(page_title="Wine Clustering")  # Wide layout

# Load the KMeans model
kmeans = pickle.load(open('wine_clustering.pkl', 'rb'))
# Streamlit app title
st.title("Wine Clustering App")

# Function to predict cluster for new data
def predict_cluster(input_text, kmeans_model):
    # Convert input text to a numpy array of float values
    input_array = np.array(input_text.split(','), dtype=np.float64)
    input_array = input_array.reshape(1, -1)  # Reshape to match model input

    # Predict cluster
    cluster = kmeans_model.predict(input_array)

    return cluster[0]

labels = [
    "Alcohol", "Malic Acid", "Ash", "Ash Alcanity", "Magnesium",
    "Total Phenols", "Flavanoids", "Nonflavanoid Phenols",
    "Proanthocyanins", "Color Intensity", "Hue", "OD280", "Proline"
]

# Concatenate the labels into a single string with commas
placeholder_text = ",".join(labels)

# Input field for comma-separated values with clear instructions
input_text = st.text_input(f"Enter comma-separated floating point values, each corresponding to:\n\n{placeholder_text}")



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
def update_plot(input_text, kmeans_model, pca_model, fig):
    global cluster, hasInput
    if input_text:
        # Predict cluster for new data
        cluster = predict_cluster(input_text, kmeans_model)
        hasInput = True
        # Perform PCA on new data point
        new_data = np.array(input_text.split(','), dtype=np.float64).reshape(1, -1)
        reduced_new_data = pca_model.transform(new_data)

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
    else:
        hasInput = False;
        

# Button to trigger prediction and plot update
if st.button("Predict Cluster"):
    update_plot(input_text, kmeans, pca, fig)


st.plotly_chart(fig)

if cluster is not None:
    st.write("Predicted Cluster: ", cluster)

if hasInput == False:
    st.write("Please enter values separated by commas.")