import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train_s, y_train)


y_pred = model.predict(X_test_s)


results = pd.DataFrame({
    "Actual":    [data.target_names[i] for i in y_test],
    "Predicted": [data.target_names[i] for i in y_pred]
})
print("Predictions on test set:\n")
print(results.to_string(index=False))


cm      = confusion_matrix(y_test, y_pred)
cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=data.target_names,
            yticklabels=data.target_names,
            ax=axes[0], cbar=False)
axes[0].set_title("Confusion Matrix (Counts)")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")


sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens",
            xticklabels=data.target_names,
            yticklabels=data.target_names,
            ax=axes[1], cbar=False)
axes[1].set_title("Confusion Matrix (Normalized)")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()