import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

with open('modelo_treinado.pkl', 'rb') as f:
    clf = pickle.load(f)

input_image_path = 'input_image7.png'
input_image = Image.open(input_image_path).convert('L')
input_image = np.array(input_image.resize((8, 8)))
input_image = input_image.reshape(1, -1)

prediction = clf.predict(input_image)

plt.imshow(input_image.reshape(8, 8), cmap='gray')
plt.title(f"Predição: {prediction[0]}")
plt.axis('off')
plt.show()