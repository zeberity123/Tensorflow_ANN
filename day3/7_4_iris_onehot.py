import pandas as pd
import numpy as np
import keras
from sklearn import preprocessing

def read_iris():
    data = pd.read_csv('data/iris_onehot.csv')
    return data.values[:, :-3], data.values[:, -3:]

x, y = read_iris()
x = preprocessing.minmax_scale(x) # 정규화(normalization)
x = preprocessing.scale(x) # 표준화(normalization)

indices = np.arange(len(x))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

n_data = int(x.shape[0]*0.7)
x_train, x_test = x[:n_data], x[n_data:]
y_train, y_test = y[:n_data], y[n_data:]

model = keras.Sequential()
model.add(keras.layers.Dense(3, activation='softmax'))
model.compile(optimizer=keras.optimizers.SGD(1),
              loss=keras.losses.categorical_crossentropy,
              metrics='acc')

model.fit(x_train, y_train, epochs=50, verbose=2)

p = model.predict(x_test, verbose=2)

# print(p)

p_arg = np.argmax(p, axis=1)
y_arg = np.argmax(y_test, axis=1)
print(p_arg)
print(y_arg)

print('acc: ', np.mean(p_arg == y_arg))