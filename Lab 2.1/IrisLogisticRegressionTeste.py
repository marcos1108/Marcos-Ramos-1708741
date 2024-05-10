import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import joblib

# Carregar os dados de teste e informações adicionais
X_test, y_test, feature_names, target_names = joblib.load('dados_teste.pkl')

# Carregar o modelo treinado e o PCA(que não será usado nesse código)
logistic_regression_model, pca = joblib.load('modelo_log_reg_com_pca.pkl')

# Avaliar o modelo
y_predicted = logistic_regression_model.predict(X_test)
accuracy = accuracy_score(y_test, y_predicted)
print("Precisão do modelo:", accuracy)

# Plotar as fronteiras de decisão
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Definir a resolução da malha
mesh_resolution = .02
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_resolution), np.arange(y_min, y_max, mesh_resolution))
Z = logistic_regression_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Configurações do gráfico
plt.figure(figsize=(12, 5))

# Plotar as fronteiras de decisão com os dados reais
plt.subplot(1, 2, 1)
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', s=20)
for i, target in enumerate(target_names):
    plt.text(X_test[y_test == i, 0].mean(), X_test[y_test == i, 1].mean(), target, color='black', fontsize=12)
plt.title("Dados Reais e Fronteira de Decisão")
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])

# Plotar as fronteiras de decisão com os dados previstos
plt.subplot(1, 2, 2)
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_predicted, cmap=cmap_bold, edgecolor='k', s=20)
for i, target in enumerate(target_names):
    plt.text(X_test[y_predicted == i, 0].mean(), X_test[y_predicted == i, 1].mean(), target, color='black', fontsize=12)
plt.title("Dados Previstos e Fronteira de Decisão")
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])

plt.tight_layout()
plt.show()
