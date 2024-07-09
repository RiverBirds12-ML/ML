import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 讀取MNIST數據集
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"].astype(np.uint8)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化特徵
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 邏輯回歸
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# 帶L2修正的邏輯回歸 (Ridge Classifier)
ridge_clf = RidgeClassifier()
ridge_clf.fit(X_train, y_train)
y_pred_ridge = ridge_clf.predict(X_test)

# 計算準確率
acc_log = accuracy_score(y_test, y_pred_log)
acc_ridge = accuracy_score(y_test, y_pred_ridge)

# 結果可視化
labels = ['Logistic Regression', 'Ridge Classifier']
accuracies = [acc_log, acc_ridge]

plt.bar(labels, accuracies, color=['blue', 'green'])
plt.ylabel('Accuracy')
plt.title('Comparison of Logistic Regression and Ridge Classifier on MNIST Dataset')
plt.ylim([0.9, 1])
plt.show()

print(f"Logistic Regression Accuracy: {acc_log:.4f}")
print(f"Ridge Classifier Accuracy: {acc_ridge:.4f}")
