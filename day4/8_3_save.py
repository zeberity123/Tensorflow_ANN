import numpy as np
from sklearn import preprocessing, model_selection
import pandas as pd
import keras

def save_model(model_path):
    mnist = keras.datasets.mnist.load_data()

    (x_train, y_train), (x_test, y_test) = mnist

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    x_train = x_train / 255
    x_test = x_test / 255

    model = keras.Sequential([
        keras.layers.Dense(10, activation='softmax')
    ])
    # model.summary()

    model.compile(optimizer=keras.optimizers.RMSprop(0.001),
                loss=keras.losses.sparse_categorical_crossentropy,
                metrics='acc')

    model.fit(x_train, y_train, epochs=10, verbose=2,
            batch_size=100)

    print('acc :', model.evaluate(x_test, y_test, verbose=2))

    model.save(model_path)

def load_model(model_path):
    mnist = keras.datasets.mnist.load_data()

    (_, _), (x_test, y_test) = mnist

    x_test = x_test.reshape(-1, 784)
    x_test = x_test / 255

    model = keras.models.load_model(model_path)

    print('acc :', model.evaluate(x_test, y_test, verbose=2))

model_path = 'model/mnist.keras'
save_model(model_path)
load_model(model_path)