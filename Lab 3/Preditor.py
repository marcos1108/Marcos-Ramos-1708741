import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import NearestNeighbors

# Carregar o modelo treinado e os parâmetros salvos
kmeans, pca, tsne, data_pca, data_tsne, cluster_to_digit, scaler = joblib.load('modelo_kmeans.joblib')

# Caminho da imagem
image_path = 'input_image.png'

# Carregar e preprocessar a imagem
img = Image.open(image_path).convert('L')  # Converter para escala de cinza
img = np.array(img)  # Converter para array numpy
img = 16 - (img / 16)  # Inverter a imagem (opcional, dependendo da sua imagem original)
img = img.flatten()  # Achatar o array para 1D

# Normalizar a imagem
image_data_normalized = scaler.transform([img])

# Transformar a imagem usando PCA
image_pca = pca.transform(image_data_normalized)

# Aproximar a transformação t-SNE usando vizinhos mais próximos no espaço PCA
nbrs = NearestNeighbors(n_neighbors=5).fit(data_pca)
distances, indices = nbrs.kneighbors(image_pca)

# Obter a transformação t-SNE aproximada para a nova imagem
image_tsne = np.mean(data_tsne[indices[0]], axis=0).reshape(1, -1)

# Fazer a previsão usando o modelo K-means
cluster_label = kmeans.predict(image_tsne)[0]
predicted_digit = cluster_to_digit[cluster_label]

print(f'O dígito previsto é: {predicted_digit}')

# Mostrar a imagem original
plt.imshow(np.reshape(img, (8, 8)), cmap='gray')
plt.title(f'Previsão: {predicted_digit}')

# Salvar a imagem
output_image_path = 'output_image.png'
plt.savefig(output_image_path)
plt.show()

print(f"A imagem foi salva como {output_image_path}")
