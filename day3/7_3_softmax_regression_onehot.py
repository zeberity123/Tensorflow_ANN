import keras
import numpy as np


x = [[1, 2],
     [2, 1],
     [4, 5],
     [5, 4],
     [8, 9],
     [9, 8]]

y = [[0, 0, 1],
     [0, 0, 1],
     [0, 1, 0],
     [0, 1, 0],
     [1, 0, 0],
     [1, 0, 0]]

model = keras.Sequential()
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.categorical_crossentropy,
              metrics='acc')

model.fit(x, y, epochs=10, verbose=2)

p = model.predict(x, verbose=0)
print(p)

p_arg = np.argmax(p, axis=1)
y_arg = np.argmax(y, axis=1)
print(p_arg)
print(y_arg)

print('acc: ', np.mean(p_arg == y_arg))