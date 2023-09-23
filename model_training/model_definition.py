import tensorflow as tf

from tensorflow.keras import layers

def classifier_sublayer(prev_layer, string_id, number):
    x = layers.Conv1D(filters=16, kernel_size=64, padding="same", name=f"{string_id}_conv1d_{number}")(prev_layer)
    x = layers.BatchNormalization(name=f"{string_id}_batch_normalization_{number}")(x)
    return layers.ReLU(name=f"{string_id}_relu_{number}")(x)

def classify_timeseries(prev_layer, string_id):
    x = classifier_sublayer(prev_layer, string_id, 1)
    x = classifier_sublayer(x, string_id, 2)

    x = layers.GlobalAveragePooling1D(name=string_id + "_global_average_pooling_1d")(x)

    return x

def uncompiled_model():

    input_shape = (1024,1,)

    i_input = tf.keras.Input(shape=input_shape, name="I")
    q_input = tf.keras.Input(shape=input_shape, name="Q")

    inputs = [i_input, q_input]

    i = classify_timeseries(i_input, 'I')
    q = classify_timeseries(q_input, 'Q')

    combined = layers.Average(name="iq_average")([i, q])

    symbol_rate_output = layers.Dense(8, activation="softmax", name="symbol_rate")(combined)

    outputs = [symbol_rate_output]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

def compiled_model():
    model = uncompiled_model()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer="adam",
                  loss=[loss_fn],
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model

def generate_image():
    model = uncompiled_model()

    tf.keras.utils.plot_model(model)
