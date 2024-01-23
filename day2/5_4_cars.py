# 5_4_cars.py
import pandas as pd
import keras
import matplotlib.pyplot as plt


# 퀴즈
# cars.csv 파일에 대해 딥러닝 모델을 구축하세요
# 속도가 30과 50일 때의 제동거리를 구하세요
def read_cars():
    # 1번
    # cars = pd.read_csv('data/cars.csv')
    # print(cars)
    # print(cars.values)

    # x = cars.values[:, 1:-1]
    # y = cars.values[:, -1:]
    # print(x.shape, y.shape)

    # 2번
    cars = pd.read_csv('data/cars.csv', index_col=0)
    print(cars)

    x = cars.speed.values.reshape(-1, 1)
    y = cars['dist'].values.reshape(-1, 1)
    print(x.shape, y.shape)

    return x, y


x, y = read_cars()

model = keras.Sequential()          # functional
model.add(keras.layers.Dense(1))

model.compile(optimizer=keras.optimizers.SGD(0.0001),
              loss=keras.losses.mse)

model.fit(x, y, epochs=10, verbose=2)   # 0, 1, 2

p = model.predict([[0], [30], [50]], verbose=0)
print(p)
p0, p1, p2 = p[0, 0], p[1, 0], p[2, 0]
plt.plot(x, y, 'ro')
plt.plot([0, 30, 50], [p0, p1, p2])
plt.show()