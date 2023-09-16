import tensorflow as tf
import keras

from keras import layers

model = keras.Sequential()

model.add(keras.Input(shape=(1024, 1)))
model.add(layers.Conv1D(32, 5, strides=2, activation="relu"))
model.add(layers.Conv1D(32, 3, activation="relu"))

model.add(layers.MaxPooling1D(5))

model.add(layers.Conv1D(32, 3, activation="relu"))
model.add(layers.Conv1D(32, 3, activation="relu"))
model.add(layers.MaxPooling1D(4))

model.add(layers.Conv1D(32, 3, activation="relu"))
model.add(layers.Conv1D(32, 3, activation="relu"))
model.add(layers.MaxPooling1D(3))

model.add(layers.GlobalMaxPooling1D())

model.add(layers.Dense(2))

model.compile(optimizer = keras.optimizers.RMSprop(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
