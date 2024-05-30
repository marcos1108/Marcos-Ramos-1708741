import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import joblib
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors

# Carregar os dados de teste, o modelo treinado e seus parâmetros
X_test, y_test = joblib.load('dados_teste.joblib')
kmeans, pca, tsne, data_pca, data_tsne, cluster_to_digit, scaler = joblib.load('modelo_kmeans.joblib')

# Transformar os dados de teste usando PCA
X_test_pca = pca.transform(X_test)

# Aproximar a transformação t-SNE usando vizinhos mais próximos no espaço PCA
nbrs = NearestNeighbors(n_neighbors=5).fit(data_pca)
distances, indices = nbrs.kneighbors(X_test_pca)

# Obter a transformação t-SNE aproximada para os dados de teste
X_test_tsne = np.array([np.mean(data_tsne[indices[i]], axis=0) for i in range(len(X_test_pca))])

# Predizer os clusters dos dados de teste
y_pred = kmeans.predict(X_test_tsne)

# Mapear clusters para dígitos usando o mapeamento salvo
y_pred_digits = np.array([cluster_to_digit[label] for label in y_pred])

# Calcular a precisão
cm = confusion_matrix(y_test, y_pred_digits)
row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
accuracy = cm[row_ind, col_ind].sum() / cm.sum()
print(f'Precisão: {accuracy:.2f}')

# Plotagem
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Subplotagem 1: Clusters identificados
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
scatter = axes[0].scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c=[colors[cluster_to_digit[label]] for label in y_pred],
                          s=50)
axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=100, c='red', linewidths=3, label='Centroides')
axes[0].set_title('Classificação dos Dados Identificados pelo K-means e Centroides')

# Adicionar a legenda dos clusters
legend_labels = [f'{i}' for i in range(10)]
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(10)]
handles.append(plt.Line2D([0], [0], marker='x', color='r', markersize=10, linestyle='None'))
legend_labels.append('Centroides')
axes[0].legend(handles, legend_labels, title="Dígitos")

# Subplotagem 2: Dados de teste classificados
scatter = axes[1].scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c=[colors[digit] for digit in y_test], s=50)
axes[1].set_title('Classificação Original dos Dados de Teste')

# Adicionar a legenda dos dígitos
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(10)]
handles.append(plt.Line2D([0], [0], marker='x', color='r', markersize=10, linestyle='None'))
axes[1].legend(handles, legend_labels, title="Dígitos")

# Ajustar layout e salvar a figura
plt.tight_layout()
plt.figtext(0.5, 0.01, f'Precisão: {accuracy:.2f}', ha='center', fontsize=12, color='black')
plt.savefig('clusters_test_accuracy.png')
plt.show()
