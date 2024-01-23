# 5_3_linear_regression.py
import keras

x = [[1],
     [2],
     [3]]
y = [[1],
     [2],
     [3]]

model = keras.Sequential()          # functional
model.add(keras.layers.Dense(1))

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.mse)

model.fit(x, y, epochs=5)
print(model.evaluate(x, y))

# 퀴즈
# x가 5와 7일 때의 결과를 구하세요
print(model.predict(x))


