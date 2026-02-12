import numpy as np

# -------------------------------
# 1) Example dataset using arrays
# -------------------------------
# Features: [sepal_length, sepal_width, petal_length, petal_width]
X = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [5.0, 3.4, 1.5, 0.2],
    [6.4, 3.2, 4.5, 1.5],
    [6.9, 3.1, 4.9, 1.5],
    [5.5, 2.3, 4.0, 1.3],
    [6.5, 3.0, 5.2, 2.0],
    [6.2, 3.4, 5.4, 2.3],
    [5.9, 3.0, 5.1, 1.8]
])

# Labels: 0=setosa, 1=versicolor, 2=virginica (example)
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# -------------------------------
# 2) Train-Test Split (manual)
# -------------------------------
np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)

split = int(0.7 * len(X))  # 70% train, 30% test
train_idx = indices[:split]
test_idx  = indices[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# -------------------------------
# 3) Euclidean Distance Function
# -------------------------------
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# -------------------------------
# 4) KNN Predict Function
# -------------------------------
def knn_predict(X_train, y_train, test_point, k=3):
    distances = []

    # Calculate distance from test_point to each training point
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], test_point)
        distances.append((dist, y_train[i]))

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Take k nearest neighbors
    k_nearest = distances[:k]

    # Vote (majority class)
    labels = [label for (_, label) in k_nearest]
    prediction = max(set(labels), key=labels.count)

    return prediction

# -------------------------------
# 5) Predict for all test samples
# -------------------------------
k = 3
predictions = []

for test_point in X_test:
    pred = knn_predict(X_train, y_train, test_point, k)
    predictions.append(pred)

predictions = np.array(predictions)

# -------------------------------
# 6) Accuracy Calculation
# -------------------------------
accuracy = np.mean(predictions == y_test) * 100

print("✅ Predictions:", predictions)
print("✅ Actual     :", y_test)
print(f"✅ Accuracy   : {accuracy:.2f}%")