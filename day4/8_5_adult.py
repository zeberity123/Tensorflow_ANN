# adult.data 파일에 대해 모델, 30%결과예측
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
import keras

def read_adult():
    data = pd.read_csv('data/adult.data', header=None)
    x = [
        data[0].values,
        data[2].values,
        data[4].values,
        data[10].values,
        data[11].values,
        data[12].values,
    ]
    x = np.float32(x).transpose()
    print(x.shape)

    bin = preprocessing.LabelBinarizer()
    y = bin.fit_transform(data[14])

    work = bin.fit_transform(data[1])
    edu = bin.fit_transform(data[3])
    marital = bin.fit_transform(data[5])
    occu = bin.fit_transform(data[6])
    rel = bin.fit_transform(data[7])
    race = bin.fit_transform(data[8])
    sex = bin.fit_transform(data[9])
    country = bin.fit_transform(data[13])

    print(work)
    print(work.shape)

    x = np.hstack([x, work, edu, marital, occu, rel, race, sex, country])
    print(x.shape)

    # exit()


    return x, y

x, y = read_adult()

data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test, y_train, y_test = data

model = keras.Sequential()
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.binary_crossentropy,
              metrics='acc')

model.fit(x_train, y_train, epochs=10, verbose=2,
          validation_data=(x_test, y_test))
