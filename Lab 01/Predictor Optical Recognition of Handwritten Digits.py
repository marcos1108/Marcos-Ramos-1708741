import pickle as p1
import pandas as pd

data_x = input("introduza valores para teste\n")
data = data_x.split(",")
print(data)
fmap_data = map(float, data)
print(fmap_data)
flist_data = list(fmap_data)
print(flist_data)
data1 = pd.read_csv("datasets/optdigits.tes", sep=",", header=None)
data2 = data1.iloc[:0, :64]
data_preparation = pd.DataFrame([flist_data], columns=list(data2))
out = data2
for x in out:
    print(x, data_preparation[x].values)
loaded_model = p1.load(open('optical_recognition_of_handwritten_digits_predictor.pkl', 'rb'))
y_pred = loaded_model.predict(data_preparation)
y_pred = y_pred.astype(int)
print("Optical Recognition of Handwritten:", y_pred[0][0])
