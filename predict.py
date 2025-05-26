# Calling our Trained Model to cluster the fruits with test data.

# Importing Libraries
import pickle
import pandas as pd
import numpy as np
from collections import Counter

# Load Model
with open('model/fruit_clustering.pkl', 'rb') as f:
    model = pickle.load(f)

# Load Training Data From CSV File
training_data = pd.read_csv('data\\training_fruits.csv', usecols= ['fruit'])

# For each cluster, find the most common fruit
cluster_to_fruit = {}

for cluster_id in np.unique(model.labels_):
    # Get index location where cluster == cluster_id
    idx = np.where(model.labels_ == cluster_id)[0]

    # Fruits in that cluster
    fruits_in_cluster = training_data['fruit'].iloc[idx]

    # Get the most common fruit in this cluster
    most_common = Counter(fruits_in_cluster).most_common(1)[0][0]

    cluster_to_fruit[cluster_id] = most_common

# Load Test Data for Prediction by Model
test_data = pd.read_csv('data\\test_fruits.csv', usecols= ['weight','color_score','length','diameter','firmness'])

# Perform PCA 
from principle_component_analysis import PCA 

pca = PCA(n_components=2) 
pca.fit(test_data)
test_data_transformed = pca.transform(test_data)

# Pass Test Data for Prediction 
labels = model.predict(test_data_transformed) # predict returns labels: Index of the cluster each sample belongs to.
labels_fruit = []

for i in labels:
    labels_fruit.append(cluster_to_fruit[i])

# Save in a new Test Data results file
# pd.DataFrame['fruit'] = labels_fruit

print(labels_fruit)