import tensorflow as tf
import keras
import json

from keras import layers

def uncompiled_model():

    inputs = keras.Input(shape=(1024, 2), name="iq")

    x = layers.Conv1D(32, 5, strides=2, activation="relu")(inputs)
    x = layers.Conv1D(32, 3, activation="relu")(x)

    x = layers.MaxPooling1D(5)(x)

    x = layers.Conv1D(32, 3, activation="relu")(x)
    x = layers.Conv1D(32, 3, activation="relu")(x)
    x = layers.MaxPooling1D(4)(x)

    x = layers.Conv1D(32, 3, activation="relu")(x)
    x = layers.Conv1D(32, 3, activation="relu")(x)
    x = layers.MaxPooling1D(3)(x)

    x = layers.GlobalMaxPooling1D()(x)

    symbol_rate_output = layers.Dense(7, name="symbol_rate")(x)
    presence_output = layers.Dense(2, name="signal_present")(x)

    outputs = [presence_output]

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

    return model

def compiled_model():
    model = uncompiled_model()

    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer = keras.optimizers.RMSprop(),
                  loss=[loss_fn],
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])

    return model

data_shape = (1024, 1)
scalar = (1)

features = {"I":              tf.io.FixedLenFeature(data_shape, tf.float32),
            "Q":              tf.io.FixedLenFeature(data_shape, tf.float32),
            "signal_present": tf.io.FixedLenFeature(scalar, tf.int64),
            "symbol_rate":    tf.io.FixedLenFeature(scalar, tf.int64),
            "phase_offset":   tf.io.FixedLenFeature(scalar, tf.float32),
            "time_offset":    tf.io.FixedLenFeature(scalar, tf.int64),
            "snr":            tf.io.FixedLenFeature(scalar, tf.float32)};

raw_dataset = tf.data.TFRecordDataset(filenames="../training_data/training_data.tfrec")

def decode_record(record_bytes):
    return tf.io.parse_single_example(record_bytes, features)

dataset = raw_dataset.map(decode_record)

def reshape_data(example):

    print(example)

    return ({
                'iq': tf.concat([example['I'], example['Q']], 1)
            },
            {
                'signal_present': example['signal_present'],
                #'symbol_rate':    example['symbol_rate'],
                #'phase_offset':   example['phase_offset'],
                #'time_offset':    example['time_offset'],
                #'snr':            example['snr']
            })

print(dataset)

dataset = dataset.map(reshape_data)

train_dataset = dataset.shuffle(buffer_size=1024).batch(64)

model = compiled_model()

model.fit(train_dataset, epochs=50)
