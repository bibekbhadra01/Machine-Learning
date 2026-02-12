import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1) Load iris.data dataset
# -----------------------------
col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv("iris.data", header=None, names=col_names)

df.dropna(inplace=True)

# -----------------------------
# 2) Define X (features) and y (target)
# -----------------------------
X = df[["sepal_length", "sepal_width", "petal_width"]].values
y = df["petal_length"].values.reshape(-1, 1)

# -----------------------------
# 3) Feature Scaling (important for GD)
# Standardization: (x - mean) / std
# -----------------------------
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# -----------------------------
# 4) Add Bias column (x0 = 1)
# -----------------------------
m = X.shape[0]  # number of samples
X = np.c_[np.ones((m, 1)), X]  # shape = (m, n+1)

# -----------------------------
# 5) Initialize Weights
# -----------------------------
n = X.shape[1]  # number of features including bias
theta = np.zeros((n, 1))  # weights

# -----------------------------
# 6) Gradient Descent Function
# -----------------------------
def gradient_descent(X, y, theta, learning_rate=0.01, epochs=1000):
    m = len(y)
    cost_history = []

    for i in range(epochs):
        # Prediction
        y_pred = X.dot(theta)

        # Error
        error = y_pred - y

        # Gradient
        gradient = (1 / m) * (X.T.dot(error))

        # Update theta
        theta = theta - learning_rate * gradient

        # Cost function (MSE/2)
        cost = (1 / (2 * m)) * np.sum(error ** 2)
        cost_history.append(cost)

        # Print sometimes
        if i % 100 == 0:
            print(f"Epoch {i} | Cost: {cost:.6f}")

    return theta, cost_history

# -----------------------------
# 7) Train the model
# -----------------------------
theta, cost_history = gradient_descent(X, y, theta, learning_rate=0.01, epochs=2000)

print("\n✅ Final Weights (theta):")
print(theta)

# -----------------------------
# 8) Final Prediction
# -----------------------------
y_pred = X.dot(theta)

# -----------------------------
# 9) Evaluation Metrics
# -----------------------------
mse = np.mean((y_pred - y) ** 2)
mae = np.mean(np.abs(y_pred - y))

# R2 Score
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
r2 = 1 - (ss_res / ss_total)

print("\n✅ Evaluation:")
print("MSE:", mse)
print("MAE:", mae)
print("R2 :", r2)

# -----------------------------
# 10) Plot Cost vs Epochs
# -----------------------------
plt.plot(cost_history)
plt.xlabel("Epochs")
plt.ylabel("Cost (MSE/2)")
plt.title("Gradient Descent Cost Function Curve")
plt.show()