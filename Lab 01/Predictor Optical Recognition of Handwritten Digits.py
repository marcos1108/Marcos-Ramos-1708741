import pickle as p1
import pandas as pd

loaded_model = p1.load(open('optical_recognition_of_handwritten_digits_predictor.pkl', 'rb'))
data_x = input("Introduza valores para teste separados por vírgula:\n")
data = data_x.split(",")
flist_data = list(map(int, data))
data_preparation = pd.DataFrame([flist_data])
y_pred = loaded_model.predict(data_preparation)
y_pred = y_pred.astype(int)
print("Optical Recognition of Handwritten:", y_pred[0])

