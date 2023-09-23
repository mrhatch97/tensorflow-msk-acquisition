import tensorflow as tf
import json
import sys

from google.protobuf import json_format

if len(sys.argv) < 2:
    print("Error: Expected path to input JSON file as first argument", file=sys.stderr)

    sys.exit(2)

data_file_path = sys.argv[1]

with open(data_file_path, 'r') as file:
    json_string = file.read()

json_examples = json_string.splitlines()

examples = []

for json_example in json_examples:
    examples.append(json_format.Parse(json_example, tf.train.Example()))

if len(sys.argv) >= 3:
    out_file_path = sys.argv[2]

else:
    out_file_path = data_file_path.replace('.json', '.tfrec')

with tf.io.TFRecordWriter(out_file_path) as writer:
    for example in examples:
        writer.write(example.SerializeToString())
