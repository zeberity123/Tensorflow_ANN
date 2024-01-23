import pandas as pd
import keras
import matplotlib

def read_cars():
    cars = pd.read_csv('data/cars.csv', index_col=0)
    return cars.values[:, :-1], cars.values[:, -1:]

# 속도가 30과 50일 때의 제동거리
x, y = read_cars()
print(x, y)
model = keras.Sequential()
model.add(keras.layers.Dense(1))

model.compile(optimizer=keras.optimizers.SGD(0.00007),
              loss=keras.losses.mse)
model.fit(x, y, epochs=5, verbose=2)
print(model.predict([[30], [50]]))