import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import mplcursors  # Import mplcursors library for hover tooltips

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

# Input field for comma-separated values
input_text = st.text_input("Enter comma-separated values")

# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter([], [], cmap='viridis')

# Perform PCA on existing data
data = pd.read_csv("wine-clustering.csv")
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

# Plot existing data points with cluster colors
existing_scatter = []
for i in range(kmeans.n_clusters):
    cluster_color = plt.cm.viridis(i / kmeans.n_clusters)  # Get cluster color from colormap
    scatter = ax.scatter(reduced_data[kmeans.labels_ == i, 0], reduced_data[kmeans.labels_ == i, 1], label=f'Cluster {i}', color=cluster_color)
    existing_scatter.append(scatter)

# Plot the "New Data" label
new_data_label = ax.scatter([], [], c='red', marker='*', s=200, label='')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('Wine Clustering (PCA)')
cluster = None
empty = False
# Button to trigger prediction
if st.button("Predict Cluster"):
    if input_text:
        # Predict cluster for new data
        cluster = predict_cluster(input_text, kmeans)
        # Perform PCA on new data point
        new_data = np.array(input_text.split(','), dtype=np.float64).reshape(1, -1)
        reduced_new_data = pca.transform(new_data)

        # Highlight newly predicted data point
        ax.scatter(reduced_new_data[:, 0], reduced_new_data[:, 1], c='red', marker='*', s=200, label='New Data')

        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('Wine Clustering (PCA)')
    else:
        empty = True


# Update the legend to include both existing scatter and "New Data" label
handles, labels = ax.get_legend_handles_labels()
handles = existing_scatter + [new_data_label]
labels = [f'Cluster {i}' for i in range(kmeans.n_clusters)] + ['New Data']
ax.legend(handles, labels)

# Display the plot
st.pyplot(fig)
if cluster != None:
    st.write(f"Predicted Cluster: {cluster}")
if empty:
    st.write(f"Please input 13 features separated by comma")