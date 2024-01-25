# car.data 파일에 대해 동작모델
# 70%로 학습, 30%로 결과

import numpy as np
from sklearn import preprocessing, model_selection
import pandas as pd
import keras

def read_cars():
    data = pd.read_csv('data/car.data')

    enc = preprocessing.LabelEncoder()
    # features = []
    # for i in range(6):
    #     result = enc.fit_transform(data.values[:, i])
    #     # print(result)
    #     features.append(result)

    # x = np.int32(features)
    # x = x.transpose()
    # # x = x.T
    # # print(x.shape)

    # y = enc.fit_transform(data.values[:, -1])

    # return x, y
    features = [enc.fit_transform(data.values[:, i]) for i in range(6)]
    return np.int32(features).T, enc.fit_transform(data.values[:, -1])

x, y = read_cars()
# x = preprocessing.minmax_scale(x) # 정규화(normalization)
# x = preprocessing.scale(x) # 표준화(normalization)

data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test, y_train, y_test = data

model = keras.Sequential()
model.add(keras.layers.Dense(4, activation='softmax'))

model.compile(optimizer=keras.optimizers.SGD(0.04),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics='acc')

model.fit(x_train, y_train, epochs=100, verbose=2,
          validation_data=(x_test, y_test))

# model.fit(x, y, epochs=50, verbose=2,
#           validation_split=0.3)

p = model.predict(x_test, verbose=0)

p_arg = np.argmax(p, axis=1)
# print(p_arg)

print('acc: ', np.mean(p_arg == y_test))