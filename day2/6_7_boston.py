import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
import keras

def read_boston():
    data = pd.read_excel('data/boston.xls')
    return data.values[:, :-2], data.values[:, -2:-1]

x, y = read_boston()
x = preprocessing.minmax_scale(x) # 정규화(normalization)
x = preprocessing.scale(x) # 표준화(normalization)

data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test, y_train, y_test = data

model = keras.Sequential()
model.add(keras.layers.Dense(1))

model.compile(optimizer=keras.optimizers.SGD(0.01),
              loss=keras.losses.mse,
              metrics='mae')

model.fit(x_train, y_train, epochs=50, verbose=2)
print('mae: ', model.evaluate(x_test, y_test, verbose=0))

p = model.predict(x_test, verbose=0)
# print('acc: ', np.mean((p > 0.5) == y_test))
print('mae: ', np.mean(np.abs(p-y_test)))