import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('data/kc_house_data.csv')
print(df.head())

feature = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot']
X = df.loc[:, feature]
Y = df.loc[:, 'price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
accuracy = []
accuracy_random = []

for n in range(10,200,10):
    clt = BaggingRegressor(n_estimators=n, random_state=0)
    clt_random = RandomForestRegressor(n_estimators=n, random_state=0)
    clt.fit(X_train, Y_train)
    clt_random.fit(X_train, Y_train)
    accuracy.append(clt.score(X_test, Y_test))
    accuracy_random.append(clt_random.score(X_test, Y_test))

plt.plot(range(10,200,10), accuracy)
plt.plot(range(10,200,10), accuracy_random)
plt.legend(['Bag tree', 'Random forest'])
plt.show()