# Importing Libraries
import pandas as pd
from sklearn.cluster import KMeans
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load Training Data From CSV File
training_data = pd.read_csv('data\\training_fruits.csv', usecols= ['weight','color_score','length','diameter','firmness'])

# Prepare Data by Performing PCA for Dimensionality Reduction for K-means Clustering
from principle_component_analysis import PCA 

pca = PCA(n_components=2) 
pca.fit(training_data)
training_data_transformed = pca.transform(training_data)

x_axis = training_data_transformed[:, 0]
y_axis = training_data_transformed[:, 1]

# Visualize Data The Dimensionality Reduced Data
plt.scatter(x_axis, y_axis)
plt.show(block=False)

# Train Model
model = KMeans(n_clusters=3) # Since we are testing this on 3 fruits, we want 3 Clusters
model.fit(training_data_transformed)

# Save model
with open('model/fruit_clustering.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved!")    











