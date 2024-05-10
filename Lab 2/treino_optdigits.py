import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pickle

with open('optdigits.tra', 'r') as f:
    data_train = np.genfromtxt(f, delimiter=',')

X_train = data_train[:, :-1]
y_train = data_train[:, -1]

unique_digits = np.unique(y_train)
samples_per_digit = 1

num_cols = 4
num_rows = int(np.ceil(len(unique_digits) / num_cols))
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

for i, digit in enumerate(unique_digits):
    digit_samples = X_train[y_train == digit][:samples_per_digit]
    ax = axes[i // num_cols, i % num_cols]
    ax.imshow(digit_samples.reshape(-1, 8, 8)[0], cmap='gray')
    ax.set_title(f'Digit: {int(digit)}')
    ax.axis('off')

for i in range(len(unique_digits), num_rows * num_cols):
    fig.delaxes(axes.flatten()[i])

plt.tight_layout()
plt.show()

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

with open('modelo_treinado.pkl', 'wb') as f:
    pickle.dump(clf, f)
