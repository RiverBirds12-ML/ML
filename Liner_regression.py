import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/linear.csv")
print(df.head())

print(df.shape)
X = df.loc[:,['Feature']].values
print(X.shape)
Y = df.loc[:,['Target']].values
print(Y.shape)

reg = LinearRegression(fit_intercept=True)
reg.fit(X,Y)

score = reg.score(X,Y)
print(score)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,7))
ax.scatter(X,Y, color = 'black')
ax.plot(X,reg.predict(X),color = 'red', linewidth = 3)

fig.tight_layout
plt.show()