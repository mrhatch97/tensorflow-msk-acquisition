import tensorflow as tf
import json

from google.protobuf import json_format

data_file_path = '../training_data/training_data.json'

with open(data_file_path, 'r') as file:
    json_string = file.read()

json_examples = json_string.splitlines()

examples = []

for json_example in json_examples:
    examples.append(json_format.Parse(json_example[:-1], tf.train.Example()))

out_file_path = data_file_path.replace('.json', '.tfrec')

with tf.io.TFRecordWriter(out_file_path) as writer:
    for example in examples:
        writer.write(example.SerializeToString())
