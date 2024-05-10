import pickle as p1
import pandas as pd
from sklearn.metrics import accuracy_score

evaluation_data = pd.read_csv("datasets/optdigits.tes", sep=",", header=None)
print(evaluation_data)
data_X = evaluation_data.iloc[:, :64]
data_Y = evaluation_data.iloc[:, 64:65]

loaded_model = p1.load(open('optical_recognition_of_handwritten_digits_predictor.pkl', 'rb'))
print("Coefficients: \n", loaded_model.coef_)

y_pred = loaded_model.predict(data_X)
y_pred = y_pred.astype(int)
accuracy = accuracy_score(data_Y, y_pred)

print("Accuracy:", accuracy)
print("Error rate:", 1 - accuracy)

