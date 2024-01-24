# 7_1_breast_cancer.py

import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
import keras

def read_wdbc():
    data = pd.read_csv('data/wdbc.data')
    return data.values[:, 2:-1], data.values[:, 1:2]

def label_encoder(y_wdbc):
    enc = preprocessing.LabelEncoder()
    enc.fit(y_wdbc)
    result = enc.transform(y_wdbc)
    result = np.reshape(result, (-1, 1))

    # enc = preprocessing.LabelEncoder()
    # enc.classes_ = np.array(['M', 'B'])
    # print(enc.classes_)
    # enc.classes_ = enc.classes_[::-1]
    # print(enc.classes_)

    return result

x, ty = read_wdbc()
# print(ty)
y = label_encoder(ty)
# print(y)
x = preprocessing.minmax_scale(x) # 정규화(normalization)
x = preprocessing.scale(x) # 표준화(normalization)

data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test, y_train, y_test = data

model = keras.Sequential()
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.SGD(0.01),
              loss=keras.losses.mse,
              metrics='mae')

model.fit(x_train, y_train, epochs=50, verbose=2)
print('mae: ', model.evaluate(x_test, y_test, verbose=0))

p = model.predict(x_test, verbose=0)
# print(p)
print('acc: ', np.mean((p > 0.5) == y_test))
# print('mae: ', np.mean(np.abs(p-y_test)))