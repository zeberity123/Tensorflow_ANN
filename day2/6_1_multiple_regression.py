import keras

# 공부시간, 출석일 수
x = [[1, 2],
     [2, 1],
     [4, 5],
     [5, 4],
     [8, 9],
     [9, 8]]

y = [[3],
     [3],
     [9],
     [9],
     [17],
     [17]]

model = keras.Sequential()
model.add(keras.layers.Dense(1))

model.compile(optimizer=keras.optimizers.SGD(0.01),
              loss=keras.losses.mse)

model.fit(x, y, epochs=5, verbose=2)

print(model.predict(x))