from sklearn.datasets import load_iris          
from sklearn.model_selection import train_test_split  
from sklearn.neighbors import KNeighborsClassifier    
import numpy as np


print("  KNN Classification – Iris Dataset")


iris = load_iris()

X = iris.data    
y = iris.target  

print(f"\n Dataset Info:")
print(f"   Total samples  : {X.shape[0]}")
print(f"   Features       : {X.shape[1]}")
print(f"   Classes        : {iris.target_names.tolist()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       
    random_state=42
)

print(f"\nData Split:")
print(f"   Training samples : {len(X_train)}")
print(f"   Testing samples  : {len(X_test)}")


k = 3
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)   

print(f"\n KNN Model trained with k={k} neighbors")


y_pred = model.predict(X_test)


print(f"\nClassification Results:")
print(f"{'Flower #':<10} {'Predicted':<15} {'Actual':<15}")
print("-" * 40)

for i in range(len(y_pred)):
    predicted_flower = iris.target_names[y_pred[i]]
    actual_flower = iris.target_names[y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]]
    print(f"{i+1:<10} {predicted_flower:<15} {actual_flower:<15}")


