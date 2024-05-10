import pickle as p1
import pandas as pd
from sklearn import linear_model

data = pd.read_csv("datasets/optdigits.tra", sep=",", header=None)
data_X = data.iloc[:, :64]
data_Y = data.iloc[:,64:]

regr = linear_model.LinearRegression()
preditor_linear_model = regr.fit(data_X, data_Y)
preditor_Pickle = open('optical_recognition_of_handwritten_digits_predictor.pkl', 'wb')
print("optical_recognition_of_handwritten_digits_predictor.pkl")
p1.dump(preditor_linear_model, preditor_Pickle)
