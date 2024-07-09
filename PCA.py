import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()
print(data)
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())

feature_names = ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
X = df.loc[:, feature_names].values
Y = df.loc[:, 'target'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=0)
princal_components = pca.fit_transform(X)
princal_matrix = pd.DataFrame(princal_components, columns=['component 1', 'component 2'])
print(princal_matrix.head())

loading = pca.components_.T
loading_matrix = pd.DataFrame(loading, columns=['component 1', 'component 2'], index=feature_names)
print(loading_matrix)
print(pca.explained_variance_ratio_)