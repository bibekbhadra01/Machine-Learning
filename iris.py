# KNN implementation using iris.data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
column_names = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class"
]

data = pd.read_csv("iris.data", header=None, names=column_names)
data.dropna(inplace=True)
X = data.iloc[:, 0:4]
y = data["class"] 

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("K value:", k)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = knn.predict(sample)

print("Prediction for sample", sample, ":", prediction[0])