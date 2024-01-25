import matplotlib.pyplot as plt
import keras

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

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=10)

reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)

checkpoints = keras.callbacks.ModelCheckpoint(filepath='model/mnist_{epoch:02d}-{val_loss:.5f}.keras',
                                              save_best_only=True,
                                              verbose=1)

history = model.fit(x_train, y_train, epochs=30, verbose=2,
        batch_size=100, validation_data=(x_test, y_test),
        callbacks=[checkpoints])


# print(history.history)
# print(history.history.keys()) # ['loss', 'acc', 'val_loss', 'val_acc]

# # history에 들어있는 값 그래프로

# indices = range(1, 11)

# plt.title('loss')
# plt.plot(indices, history.history['loss'], 'r')
# plt.plot(indices, history.history['val_loss'], 'g')

# plt.figure()
# plt.title('accuracy')
# plt.plot(indices, history.history['acc'], 'r')
# plt.plot(indices, history.history['val_acc'], 'g')
# # plt.show()

