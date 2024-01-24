import pandas as pd
import numpy as np
import keras
from sklearn import preprocessing

def read_iris():
    data = pd.read_csv('data/iris.csv')
    return data.values[:, :-1], data.values[:, -1:]

def encoder(ty):
    enc = preprocessing.LabelEncoder()
    enc.fit(ty)
    result = enc.transform(ty)
    result = np.reshape(result, (1, -1))[0]
    return result

x, ty = read_iris()
y = encoder(ty)
# print(y)
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

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics='acc')

model.fit(x_train, y_train, epochs=50, verbose=2)

p = model.predict(x_test, verbose=2)

p_arg = np.argmax(p, axis=1)
print(p_arg)

print('acc: ', np.mean(p_arg == y_test))
bools = (p != y)
wrong = x[bools]

y_wrong, p_wrong = y[bools], p[bools]
print(y_wrong)
print(p_wrong)
# print(classes[y_wrong])
# print(classes[p_wrong])
# wng = [np.argmax(i, axis=1) for i in zip(p_arg, y_test)]
