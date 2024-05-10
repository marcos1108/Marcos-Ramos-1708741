import joblib  # Importa a biblioteca joblib para salvar e carregar objetos Python
import numpy as np  # Importa a biblioteca NumPy para operações numéricas
import matplotlib.pyplot as plt  # Importa a biblioteca Matplotlib para plotagem
from matplotlib.colors import ListedColormap  # Importa a função ListedColormap para criar mapas de cores
from sklearn.model_selection import train_test_split  # Importa a função train_test_split para dividir os dados em conjunto de treino e teste
from sklearn.preprocessing import StandardScaler  # Importa a classe StandardScaler para padronização de dados
from sklearn.decomposition import PCA  # Importa a classe PCA para redução de dimensionalidade
from sklearn.datasets import load_iris  # Importa o conjunto de dados Iris
from sklearn.linear_model import LogisticRegression  # Importa a classe LogisticRegression para treinar o modelo de regressão logística

# Carrega o conjunto de dados Iris
iris = load_iris()

# Divide os dados em treino (2/3) e teste (1/3)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)

# Padroniza os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reduz a dimensionalidade com PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Salva os dados de treino e teste em arquivos separados
joblib.dump((X_train_pca, y_train, iris.feature_names, iris.target_names), 'dados_treino.pkl')
joblib.dump((X_test_pca, y_test, iris.feature_names, iris.target_names), 'dados_teste.pkl')

# Treina o modelo de regressão logística
log_reg = LogisticRegression()
log_reg.fit(X_train_pca, y_train)

# Salva o modelo treinado e o PCA
joblib.dump((log_reg, pca), 'modelo_log_reg_com_pca.pkl')

# Plota os dados de treino e teste com a fronteira de decisão
h = .02  # largura da malha

# Cria mapas de cores
cmap_light = ListedColormap(['#FFDDDD', '#DDFFDD', '#DDDDFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Plota a fronteira de decisão
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])

# Coloca o resultado em um gráfico de cores
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plota os pontos de treino
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cmap_bold, s=40, marker='o', label='Treino')
# Plota os pontos de teste
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=cmap_bold, s=60, marker='x', label='Teste', alpha=0.6)

# Adiciona uma legenda personalizada
handles = []
for target_name in iris.target_names:
    handles.append(plt.scatter([], [], c=cmap_bold.colors[iris.target_names.tolist().index(target_name)], label=f'{target_name} (Treino)'))
    handles.append(plt.scatter([], [], c=cmap_bold.colors[iris.target_names.tolist().index(target_name)], marker='x', alpha=0.6, label=f'{target_name} (Teste)'))

plt.legend(handles=handles)
plt.title('Dados de Treino, Teste e a Fronteira de Decisão')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()