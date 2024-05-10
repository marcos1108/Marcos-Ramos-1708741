import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn import neighbors
import pickle

data_test = np.loadtxt('optdigits.tes', delimiter=',')
X_test = data_test[:, :-1].astype(int)
y_test = data_test[:, -1].astype(int)

with open('modelo_treinado_com_nca.pkl', 'rb') as f:
    pipeline = pickle.load(f)

predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Precisão do modelo nos dados de teste: {accuracy}")

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {predictions[i]}, True: {y_test[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)
n_neighbors = 15
h = 0.2  # step size in the mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X_pca, y_test)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap=plt.cm.coolwarm, edgecolor='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Optdigits classification com NCA (k = %i, weights = 'uniform')" % n_neighbors)

plt.tight_layout()
plt.show()
