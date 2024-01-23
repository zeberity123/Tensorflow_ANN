import keras
import numpy as np

# 공부시간, 출석일 수
x = [[1, 2],
     [2, 1],
     [4, 5],
     [5, 4],
     [8, 9],
     [9, 8]]

y = [[0],
     [0],
     [1],
     [1],
     [1],
     [1]]

model = keras.Sequential()
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation('sigmoid'))

model.compile(optimizer=keras.optimizers.SGD(1),
              loss=keras.losses.binary_crossentropy,
              metrics='acc')

model.fit(x, y, epochs=50, verbose=2)
p = (model.predict(x, verbose=0))

p_bool = p > 0.5
print(p_bool)
p_int = np.int32(p_bool)
print(p_int)

equals = (p_int == y)
print(equals)
print('acc: ', np.mean(equals))
print('acc: ', np.mean(p_bool == y))
