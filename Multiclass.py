import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv('data/modifiedDigits4Classes.csv')
print(df.shape)
pixel_colnames = df.columns[:-1]
image_values = df.loc[0, pixel_colnames].values


plt.figure(figsize=(10,2))

for idx in range (0,4):
    plt.subplot(1, 5, 1 + idx)
    image_label = df.loc[idx, 'label']
    image_values = df.loc[idx, pixel_colnames].values
    plt.imshow(image_values.reshape(8,8), cmap = 'gray')
    plt.title('Label:' + str(image_label))

X_train, X_test, Y_train, Y_test = train_test_split(df[pixel_colnames], df['label'], random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Training
clt = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0)
clt.fit(X_train, Y_train)

# Predict one data
print(X_test[0].reshape(1,-1))
print('probability', clt.predict_proba(X_test[0].reshape(1,-1)))

Y_pred = clt.predict(X_test)

cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,8))
sns.heatmap(cm, xticklabels=['0', '1', '2', '3'], yticklabels=['0', '1', '2', '3'], fmt='d', cmap='Blues', annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()









