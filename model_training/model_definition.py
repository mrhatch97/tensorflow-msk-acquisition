import keras

from keras import layers

def classify_timeseries(prev_layer):
    x = layers.Conv1D(filters=7, kernel_size=64, padding="same")(prev_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(filters=7, kernel_size=64, padding="same")(prev_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(filters=7, kernel_size=64, padding="same")(prev_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)

    return x

def uncompiled_model():

    input_shape = (1024,1,)

    i_input = keras.Input(shape=input_shape, name="I")
    q_input = keras.Input(shape=input_shape, name="Q")

    inputs = [i_input, q_input]

    i = classify_timeseries(i_input)
    q = classify_timeseries(q_input)

    combined = layers.Average()([i, q])

    symbol_rate_output = layers.Dense(8, activation="softmax", name="symbol_rate")(combined)

    outputs = [symbol_rate_output]

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

def compiled_model():
    model = uncompiled_model()

    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer="adam",
                  loss=[loss_fn, loss_fn],
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])

    return model
