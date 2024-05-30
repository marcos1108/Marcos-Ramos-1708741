import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import joblib

# Carregar o dataset
data, labels = load_digits(return_X_y=True)

# Normalizar os dados
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Dividir o dataset em dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.33, random_state=42)

# Salvar os dados de treino e teste
joblib.dump((X_train, y_train), 'dados_treino.joblib')
joblib.dump((X_test, y_test), 'dados_teste.joblib')

# Redução de dimensionalidade usando PCA
pca = PCA(n_components=50)
data_pca = pca.fit_transform(X_train)

# Redução de dimensionalidade usando t-SNE com os melhores parâmetros encontrados
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200, n_iter=1000)
data_tsne = tsne.fit_transform(data_pca)

# Treinar o modelo K-means com o número ótimo de clusters
kmeans = KMeans(n_clusters=10, init='k-means++', random_state=42)
kmeans.fit(data_tsne)
cluster_labels = np.unique(kmeans.labels_)
cluster_to_digit = {cluster: np.argmax(np.bincount(y_train[kmeans.labels_ == cluster])) for cluster in cluster_labels}

# Salvar o modelo treinado, PCA, t-SNE e os parâmetros
joblib.dump((kmeans, pca, tsne, data_pca, data_tsne, cluster_to_digit, scaler), 'modelo_kmeans.joblib')

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.delaxes(axes[0, 1])
axes[0, 0] = plt.subplot2grid((2, 2), (0, 0), colspan=2)

# Plot dos clusters para o k ótimo (Elbow Method)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
scatter = axes[0, 0].scatter(data_tsne[:, 0], data_tsne[:, 1], c=[cluster_to_digit[label] for label in kmeans.labels_],
                             s=50, cmap='tab10')
axes[0, 0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x', linewidths=3, label='Centroides')
axes[0, 0].set_title(f'Clusters for Optimal k = 10')

# Adicionar a legenda dos clusters
legend_labels = [f'{digit}' for digit in np.unique(list(cluster_to_digit.values()))]
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(10)]
handles.append(plt.Line2D([0], [0], marker='x', color='r', markersize=10, linestyle='None'))
axes[0, 0].legend(handles, legend_labels + ['Centroides'], title="Dígitos", loc="upper right", bbox_to_anchor=(1.15, 1))

# Plot do Elbow Method
axes[1, 0].plot(range(2, 21),
                [KMeans(n_clusters=k, init='k-means++', random_state=42).fit(data_tsne).inertia_ for k in range(2, 21)],
                'bx-')
axes[1, 0].set_xlabel('Valores de K')
axes[1, 0].set_ylabel('Inércia')
axes[1, 0].set_title('Método do "Cotovelo" para k Ótimo')
axes[1, 0].axvline(x=10, linestyle='--', color='black')

# Plot da análise do coeficiente de silhouette
silhouette_values = [
    silhouette_score(data_tsne, KMeans(n_clusters=k, init='k-means++', random_state=42).fit(data_tsne).labels_) for k in
    range(2, 21)]
axes[1, 1].plot(range(2, 21), silhouette_values, 'rx-')
axes[1, 1].set_xlabel('Valores de K')
axes[1, 1].set_ylabel('Pontuação de Silhueta')
axes[1, 1].set_title('Análise de Silhueta para k Ótimo')
axes[1, 1].axvline(x=10, linestyle='--', color='black')

# Ajustar layout e salvar a figura
plt.tight_layout()
plt.savefig('optimal_k_clusters_tsne.png')
plt.show()
