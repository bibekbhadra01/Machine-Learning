import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# 1. Load iris.data dataset
# ---------------------------
col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

df = pd.read_csv("iris.data", header=None, names=col_names)

# Remove empty rows (sometimes iris.data has last blank line)
df.dropna(inplace=True)

print("✅ Dataset Loaded Successfully")
print(df.head())

# ---------------------------
# 2. Split features and labels
# ---------------------------
X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 3. Define models
# ---------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

# ---------------------------
# 4. Train + Test each model
# ---------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)
    f1_macro = report["macro avg"]["f1-score"]
    precision_macro = report["macro avg"]["precision"]
    recall_macro = report["macro avg"]["recall"]

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision(Macro)": precision_macro,
        "Recall(Macro)": recall_macro,
        "F1-Score(Macro)": f1_macro
    })

    print("\n==============================")
    print(f"✅ Model: {name}")
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# 5. Create comparison dataframe
# ---------------------------
results_df = pd.DataFrame(results)

print("\n\n✅ Final Comparison Table:")
print(results_df)

# ---------------------------
# 6. LINE GRAPH (X-axis = Metrics)
# ---------------------------
metrics = ["Accuracy", "Precision(Macro)", "Recall(Macro)", "F1-Score(Macro)"]

plt.figure(figsize=(12, 6))

for i in range(len(results_df)):
    plt.plot(
        metrics,
        results_df.loc[i, metrics],
        marker="o",
        linewidth=2,
        label=results_df.loc[i, "Model"]
    )

plt.title("✅ Classification Model Comparison (Line Graph)")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()