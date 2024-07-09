import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('data/modifiedIris2Classes.csv')
print(df.columns)
print(df.shape)

# Make sure to match the column names exactly
X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = df['target']

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train,Y_train)

print(X_test[0].reshape(1,-1))
print('probability',clf.predict_proba(X_test[0].reshape(1,-1)))


Y_pred = clf.predict(X_test)
Y_prob = clf.predict_proba(X_test)[:,1]

cm = confusion_matrix(Y_test, Y_pred)

# 可視化混淆矩陣
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')


# 列印分類報告
test_data = pd.DataFrame(X_test, columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
test_data['true_class'] = Y_test.values
test_data['predicted_prob'] = Y_prob
test_data['predicted_class'] = Y_pred

# 标记错误预测的点
test_data['is_incorrect'] = test_data['true_class'] != test_data['predicted_class']

# 创建一个包含4个子图的画布
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
variables = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

for ax, var in zip(axes.flatten(), variables):
    # 按照变量的大小排序数据
    sorted_data = test_data.sort_values(by=var)
    
    sns.scatterplot(data=sorted_data, x=var, y='predicted_prob', hue='true_class', palette='coolwarm', s=100, ax=ax)
    ax.axhline(y=0.5, color='r', linestyle='--')
    ax.set_xlabel(var)
    ax.set_ylabel('Predicted Probability')
    ax.set_title(f'Predicted Probability vs {var}')
    ax.legend(title='True Class', loc='best')

# 提取變量的權重
coefficients = clf.coef_[0]
feature_names = X.columns

# 顯示變量的權重
for feature, coef in zip(feature_names, coefficients):
    print(f'{feature}: {coef}')

# 繪製變量權重的條形圖
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_names, y=coefficients)
plt.xlabel('Features')
plt.ylabel('Coefficient')
plt.title('Feature Coefficients in Logistic Regression')
plt.show()
