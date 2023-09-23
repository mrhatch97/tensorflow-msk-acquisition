import io
import keras
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

from datetime import datetime
from dataset_management import full_dataset
from symbol_rates import inverse_symbol_rate_dict
from tensorboard_config import log_dir

grid_side = 2

number_of_predictions = grid_side * grid_side

def plot_to_image(figure):
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def generate_image_grid(model, dataset):
    figure = plt.figure(figsize=(10,10), layout="constrained")

    for index, example in dataset.enumerate():
        inputs = example[0]
        targets = example[1]

        prediction = model.predict(inputs)

        actual_class = targets['symbol_rate'].numpy()[0][0]
        actual_symbol_rate = inverse_symbol_rate_dict[actual_class]

        best_class = np.argmax(prediction, axis=-1)[0]
        best_symbol_rate = inverse_symbol_rate_dict[best_class]

        confidence = prediction[0][best_class]

        i = inputs['I'].numpy().squeeze()
        q = inputs['Q'].numpy().squeeze()

        complex_input = i + 1j * q

        fft_exponentials = np.fft.fft(complex_input)
        fft_magnitudes = np.abs(fft_exponentials)
        # Convert to dB scale
        fft_spectrum = 20 * np.log10(np.fft.fftshift(fft_magnitudes))

        sample_rate_Hz = 25600
        sample_rate_kHz = sample_rate_Hz / 1000

        fft_bin_size_kHz = (sample_rate_kHz / len(fft_exponentials))

        frequencies_kHz = np.arange(-(sample_rate_kHz / 2), (sample_rate_kHz / 2), fft_bin_size_kHz)

        plt.subplot(grid_side, grid_side, index.numpy() + 1,
                    title=f"Power Spectrum, {actual_symbol_rate} sps\nPrediction: {best_symbol_rate} sps, confidence {confidence:.4f}")
        plt.plot(frequencies_kHz, fft_spectrum)
        plt.ylim(-20, 60)
        plt.xticks(np.arange(-12, 13, 2))
        plt.grid(True)

    figure.suptitle("Example Outputs", fontsize=24)
    figure.supylabel("Power (dB)", fontsize=16)
    figure.supxlabel("Frequency (kHz)", fontsize=16)

    return figure

if len(sys.argv) < 2:
    print("Error: Expected path to Keras model file as first argument", file=sys.stderr)

    sys.exit(2)

path_to_model = sys.argv[1]

model = tf.keras.models.load_model(path_to_model)

output_file = log_dir + datetime.now().strftime("%Y%m%d-%H%M%S")

file_writer = tf.summary.create_file_writer(output_file)

dataset = full_dataset()

model.evaluate(dataset.shuffle(buffer_size=1024).batch(64))

predict_dataset = dataset.shuffle(buffer_size=1024).take(number_of_predictions).batch(1)

figure = generate_image_grid(model, predict_dataset)

with file_writer.as_default():
    tf.summary.image("Example Output", plot_to_image(figure), step=0)
