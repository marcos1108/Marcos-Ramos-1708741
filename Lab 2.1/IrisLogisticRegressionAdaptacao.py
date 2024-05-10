from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Carregar o conjunto de dados Iris
iris = load_iris()
X, y = iris.data, iris.target

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33, random_state=42)

# Redução de dimensionalidade usando PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Criar e treinar o modelo de regressão logística
logreg = LogisticRegression()
logreg.fit(X_train_pca, y_train)

# Avaliar o modelo
score = logreg.score(X_test_pca, y_test)
print("Acurácia do modelo de Regressão Logística com PCA:", score)

# Visualização da classificação
XX = np.c_[X_test_pca, y_test]
XX = XX[np.argsort(XX[:, 2])]
h = 0.2  # passo no mesh
x_min, x_max = XX[:, 0].min() - 1, XX[:, 0].max() + 1
y_min, y_max = XX[:, 1].min() - 1, XX[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classificação em 3 classes com Regressão Logística e PCA")
sns.scatterplot(
    x=XX[:, 0], y=XX[:, 1], hue=iris.target_names[np.sort(y_test)],
    palette=["darkorange", "c", "darkblue"], alpha=1.0, edgecolor="black")
plt.show()
