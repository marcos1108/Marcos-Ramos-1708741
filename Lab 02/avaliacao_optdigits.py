import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn import neighbors
import pickle

# Load test data
data_test = np.loadtxt('optdigits.tes', delimiter=',')
X_test = data_test[:, :-1].astype(int)
y_test = data_test[:, -1].astype(int)

# Load trained model without NCA
with open('modelo_treinado.pkl', 'rb') as f:
    clf = pickle.load(f)

# Make predictions and calculate accuracy
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Precisão do modelo sem NCA: {accuracy}")

# Plot images with predictions
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {predictions[i]}, True: {y_test[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()

# Apply PCA for dimensionality reduction to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)

# Define number of neighbors
n_neighbors = 15

# Create meshgrid for plotting decision boundaries
h = 0.2  # step size in the mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Plot decision boundaries for uniform weight type
clf_knn = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf_knn.fit(X_pca, y_test)

Z = clf_knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot data points
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap=plt.cm.coolwarm, edgecolor='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Optdigits classification sem NCA (k = %i, weights = 'uniform')" % n_neighbors)

plt.tight_layout()
plt.show()
