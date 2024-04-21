import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle  #used to create pickle files
import joblib  #used to save models

data = pd.read_csv("wine-clustering.csv")

kmeans = KMeans(n_clusters=3, random_state=42)
data['clusters'] = kmeans.fit_predict(data)

pca_num_components = 2
reduced_data = PCA(n_components=pca_num_components).fit_transform(data)
results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])

sns.scatterplot(x="pca1", y="pca2", hue=data['clusters'], data=results)
plt.title('K-means Clustering with 2 dimensions')
plt.show()
