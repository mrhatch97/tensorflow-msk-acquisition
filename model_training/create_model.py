import tensorflow as tf
import keras
import json

from keras import layers

def feature_extraction(prev_layer):
    x = layers.Conv1D(7, 4)(prev_layer)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(7, 4)(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(7, 4)(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(7, 4)(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.GlobalMaxPooling1D()(x)

    return x

def uncompiled_model():

    i_input = keras.Input(shape=(1024, 1), name="I")
    q_input = keras.Input(shape=(1024, 1), name="Q")

    inputs = [i_input, q_input]

    i = feature_extraction(i_input)
    q = feature_extraction(q_input)

    i = layers.Dense(7)(i)
    q = layers.Dense(7)(q)

    combined = layers.Average()([i, q])

    symbol_rate_output = layers.Dense(8, name="symbol_rate")(combined)

    outputs = [symbol_rate_output]

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

    return model

def compiled_model():
    model = uncompiled_model()

    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer = keras.optimizers.RMSprop(),
                  loss=[loss_fn, loss_fn],
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])

    return model

def create_datasets():
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

    train_dataset, test_dataset = keras.utils.split_dataset(dataset, left_size=0.7)
    test_dataset, validation_dataset = keras.utils.split_dataset(test_dataset, left_size=0.5)

    # Map symbol rate to class
    symbol_rate_dict = {
        0:     0,
        4800:  1,
        9600:  2,
        14400: 3,
        16000: 4,
        19200: 5,
        24000: 6,
        28000: 7
    }
    symbol_rate_lookup = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(
            list(symbol_rate_dict.keys()),
            list(symbol_rate_dict.values()),
            key_dtype=tf.int64,
            value_dtype=tf.int64),
        num_oov_buckets=1)

    def reshape_data(example):
        return ({
                    'I': example['I'],
                    'Q': example['Q']
                },
                {
                    'symbol_rate':     symbol_rate_lookup.lookup(example['symbol_rate']),
                    #'phase_offset':   example['phase_offset'],
                    #'time_offset':    example['time_offset'],
                    #'snr':            example['snr']
                })

    train_dataset = train_dataset.map(reshape_data).shuffle(buffer_size=1024).batch(64)
    validation_dataset = validation_dataset.map(reshape_data).shuffle(buffer_size=1024).batch(64)
    test_dataset = test_dataset.map(reshape_data).shuffle(buffer_size=1024).batch(64)

    return (train_dataset, validation_dataset, test_dataset)

train_dataset, validation_dataset, test_dataset = create_datasets()

model = compiled_model()

tb_callback = keras.callbacks.TensorBoard(
    log_dir="tb_logs",
    histogram_freq=1,
    embeddings_freq=0,
    update_freq="epoch"
)

model.fit(train_dataset, epochs=15, validation_data=validation_dataset, callbacks=[tb_callback])

model.evaluate(test_dataset)
