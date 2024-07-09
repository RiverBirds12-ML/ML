import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = load_iris()
#print(data)
df = pd.DataFrame(data.data, columns=data.feature_names)
#print(df.head())
X = df.loc[:,['petal length (cm)', 'petal width (cm)']].values
Y = data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)


k_means = KMeans(n_clusters=3, random_state=0)
k_means.fit(X)
labels = k_means.labels_
centers = k_means.cluster_centers_

colormap = np.array(['r','g','b'])
plt.figure()
plt.scatter(X[:,0], X[:,1], c=colormap[labels])
plt.figure()
plt.scatter(X[:,0], X[:,1], c=colormap[Y])
plt.show()