import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import random



mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data / 255.0, mnist.target.astype(np.uint8)


X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]



knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn.fit(X_train[:10000], y_train[:10000])   


random_index = random.randint(0, len(X_test) - 1)
sample_image = X_test[random_index]

prediction = knn.predict([sample_image])[0]


plt.imshow(sample_image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted Digit: {prediction}", fontsize=16)
plt.axis('off')
plt.show()

print(f"Random index chosen: {random_index}")
print(f"Predicted digit: {prediction}")