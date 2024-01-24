# diabetes.csv 파일에 대해 딥러닝
# 70%의 데이터로 학습, 30%의 데이터로 정확도

import numpy as np
import keras
import pandas as pd
from sklearn import preprocessing, model_selection

def read_diabetes():
    data = pd.read_csv('data/diabetes.csv')
    # print(data.values[:4])
    return data.values[:, :-1], data.values[:, -1:]

x, y = read_diabetes()

x = preprocessing.minmax_scale(x) # 정규화(normalization)
x = preprocessing.scale(x) # 표준화(normalization)

data = model_selection.train_test_split(x, y, train_size=0.7)
# print(x, y)
# print(x.shape, y.shape)
t_length = int(x.shape[0] * 0.7)
# x_train, y_train, x_test, y_test = x[:t_length], y[:t_length], x[t_length:], y[t_length:]
x_train, x_test, y_train, y_test = data

# print(x_train)


model = keras.Sequential()
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.SGD(1),
              loss=keras.losses.binary_crossentropy,
              metrics='acc')

model.fit(x_train, y_train, epochs=50, verbose=2,
          validation_data=(x_test, y_test))

p = model.predict(x_test, verbose=0)
print(p)
print('acc: ', np.mean((p > 0.5) == y_test))
