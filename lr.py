import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --------------------------
# 1) Load iris.data file
# --------------------------
col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv("iris.data", header=None, names=col_names)

# Drop empty rows if any
df.dropna(inplace=True)

print(df.head())

# --------------------------
# 2) Input (X) and Target (y)
# --------------------------
X = df[["sepal_length", "sepal_width", "petal_width"]]   # features
y = df["petal_length"]                                  # numeric target

# --------------------------
# 3) Train-Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# 4) Train Linear Regression Model
# --------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------
# 5) Predict
# --------------------------
y_pred = model.predict(X_test)

# --------------------------
# 6) Evaluation
# --------------------------
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("\n✅ Linear Regression Results:")
print("MSE:", mse)
print("MAE:", mae)
print("R2 Score:", r2)

# --------------------------
# 7) Coefficients
# --------------------------
print("\n✅ Model Coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef}")

print("Intercept:", model.intercept_)