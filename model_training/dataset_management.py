import tensorflow as tf

from symbol_rates import symbol_rate_dict

training_data_path = "../training_data/training_data.tfrec"

def load_parsed_dataset():
    data_shape = (1024, 1)
    scalar = (1)

    features = {"I":              tf.io.FixedLenFeature(data_shape, tf.float32),
                "Q":              tf.io.FixedLenFeature(data_shape, tf.float32),
                "signal_present": tf.io.FixedLenFeature(scalar, tf.int64),
                "symbol_rate":    tf.io.FixedLenFeature(scalar, tf.int64),
                "phase_offset":   tf.io.FixedLenFeature(scalar, tf.float32),
                "time_offset":    tf.io.FixedLenFeature(scalar, tf.int64),
                "snr":            tf.io.FixedLenFeature(scalar, tf.float32)};

    raw_dataset = tf.data.TFRecordDataset(filenames=[training_data_path])

    def decode_record(record_bytes):
        return tf.io.parse_single_example(record_bytes, features)

    return raw_dataset.map(decode_record)

def preprocess_dataset(dataset):
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
                    'symbol_rate': symbol_rate_lookup.lookup(example['symbol_rate'])
                })

    return dataset.map(reshape_data)

def create_datasets():
    dataset = load_parsed_dataset()

    train_dataset, test_dataset = tf.keras.utils.split_dataset(dataset, left_size=0.7)
    test_dataset, validation_dataset = tf.keras.utils.split_dataset(test_dataset, left_size=0.5)

    # For some reason we have to preprocess the dataset *after* splitting, otherwise splitting doesn't work
    train_dataset = preprocess_dataset(train_dataset).shuffle(buffer_size=1024).batch(64)
    validation_dataset = preprocess_dataset(validation_dataset).shuffle(buffer_size=1024).batch(64)
    test_dataset = preprocess_dataset(test_dataset).shuffle(buffer_size=1024).batch(64)

    return (train_dataset, validation_dataset, test_dataset)

def full_dataset():
    dataset = load_parsed_dataset()

    return preprocess_dataset(dataset)
