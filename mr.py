import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# --------------------------
# 1) Load iris.data dataset
# --------------------------
col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv("iris.data", header=None, names=col_names)

# Remove empty rows if any
df.dropna(inplace=True)

print("✅ Dataset Loaded Successfully!")
print(df.head())


# --------------------------
# 2) Multiple Input Features (X) and Target (y)
# --------------------------
X = df[["sepal_length", "sepal_width", "petal_width"]]   # multiple features
y = df["petal_length"]                                  # numeric target


# --------------------------
# 3) Split into Train-Test
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --------------------------
# 4) Train Multiple Linear Regression Model
# --------------------------
model = LinearRegression()
model.fit(X_train, y_train)


# --------------------------
# 5) Predict
# --------------------------
y_pred = model.predict(X_test)


# --------------------------
# 6) Evaluate Model
# --------------------------
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n✅ Multiple Linear Regression Results:")
print("MSE  :", mse)
print("MAE  :", mae)
print("RMSE :", rmse)
print("R2   :", r2)


# --------------------------
# 7) Coefficients and Equation
# --------------------------
print("\n✅ Model Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"{col} : {coef}")

print("Intercept:", model.intercept_)

print("\n✅ Regression Equation:")
print(f"petal_length = ({model.coef_[0]:.4f} * sepal_length) + "
      f"({model.coef_[1]:.4f} * sepal_width) + "
      f"({model.coef_[2]:.4f} * petal_width) + "
      f"({model.intercept_:.4f})")


# --------------------------
# 8) Plot: Actual vs Predicted
# --------------------------
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Petal Length")
plt.ylabel("Predicted Petal Length")
plt.title("Actual vs Predicted (Multiple Linear Regression)")
plt.show()