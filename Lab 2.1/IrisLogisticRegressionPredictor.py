import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Carregar o modelo treinado e o objeto PCA
log_reg, pca = joblib.load('modelo_log_reg_com_pca.pkl')

# Carregar os dados de treino
X_train_pca, y_train, feature_names, target_names = joblib.load('dados_treino.pkl')

# Obter input do usuário
feature_values = []
for i in range(4):
    value = float(input(f"Insira o valor de {feature_names[i]}: "))
    feature_values.append(value)

# Realizar a redução de dimensionalidade
X_input_pca = pca.transform(np.array([feature_values]))

# Definir limites para a plotagem
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

# Criar a malha para plotagem
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Configurar mapa de cores
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Plotar as fronteiras de decisão com o ponto de entrada
plt.figure(figsize=(10, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plotar o ponto de entrada
plt.scatter(X_input_pca[:, 0], X_input_pca[:, 1], c='black', marker='x', label='Ponto de Entrada com PCA', s=100)

# Adicionar os nomes dos targets dentro das fronteiras de decisão
for i, target_name in enumerate(target_names):
    plt.text(X_train_pca[y_train == i, 0].mean(), X_train_pca[y_train == i, 1].mean(), target_name,
             horizontalalignment='center', verticalalignment='center', fontsize=12, color='white',
             bbox=dict(facecolor='black', alpha=0.5))

plt.title("Fronteiras de Decisão com Ponto de Entrada")
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend()
plt.show()
