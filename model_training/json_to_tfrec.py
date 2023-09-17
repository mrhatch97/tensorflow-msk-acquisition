import tensorflow as tf
import json
from google.protobuf import json_format

data_file_path = '../training_data/training_data.json'

with open(data_file_path, 'r') as file:
    json_string = file.read()

json_examples = json_string.splitlines()

data_shape = (1024, 1)
scalar = (1)

features = {"I":              tf.io.FixedLenFeature(data_shape, tf.float32),
            "Q":              tf.io.FixedLenFeature(data_shape, tf.float32),
            "signal_present": tf.io.FixedLenFeature(scalar, tf.int64),
            "symbol_rate":    tf.io.FixedLenFeature(scalar, tf.int64),
            "phase_offset":   tf.io.FixedLenFeature(scalar, tf.float32),
            "time_offset":    tf.io.FixedLenFeature(scalar, tf.int64),
            "snr":            tf.io.FixedLenFeature(scalar, tf.float32)};

examples = []

for json_example in json_examples:
    examples.append(json_format.Parse(json_example[:-1], tf.train.Example()))

out_file_path = data_file_path.replace('.json', '.tfrec')

with tf.io.TFRecordWriter(out_file_path) as writer:
    for example in examples:
        writer.write(example.SerializeToString())
