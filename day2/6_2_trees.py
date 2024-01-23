# trees.csv 파일을 읽고 Girth와 height로 volume을 예측 (10, 70), (15, 80)
import pandas as pd
import keras

def read_trees():

    trees = pd.read_csv('data/trees.csv', index_col=0)
    # print(trees)

    return trees.values[:, :-1], trees.values[:, -1:]


x, y = read_trees()
# print(x, y)
# print(x.shape, y.shape)

model = keras.Sequential()
model.add(keras.layers.Dense(1))

model.compile(optimizer=keras.optimizers.SGD(0.00007),
              loss=keras.losses.mse)

model.fit(x, y, epochs=5, verbose=2)

print(model.predict([[10, 70], [15, 80]]))
