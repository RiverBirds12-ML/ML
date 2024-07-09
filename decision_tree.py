import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree

data = load_iris()
print(data.feature_names)
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head)

X_train, X_test, Y_train, Y_test = train_test_split(df[data.feature_names],df['target'], random_state=0)

clf = DecisionTreeClassifier(max_depth=3, random_state=0)
clf.fit(X_train, Y_train)
score = clf.score(X_test, Y_test)
print(score)

Y_pred = clf.predict(X_test)

cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, cmap='Blues', fmt= 'd', xticklabels=['1','2','3'], yticklabels=['1','2','3'])
plt.xlabel('Predicted')
plt.ylabel('Actual')


fn = data.feature_names
cn = ['Setosa', 'Versicolour', 'Virginica']
print(fn)
print(cn)

plt.figure(figsize=(4,4), dpi=300)
tree.plot_tree(clf, feature_names=fn,class_names=cn, filled=True)
plt.show()

