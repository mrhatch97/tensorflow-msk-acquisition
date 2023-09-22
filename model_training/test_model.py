import keras
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from dataset_management import full_dataset
from symbol_rates import inverse_symbol_rate_dict

model = tf.keras.models.load_model("model.keras")

n_predictions = 64

predict_dataset = full_dataset().shuffle(buffer_size=1024).take(n_predictions).batch(1)

for index, example in predict_dataset.enumerate():
    inputs = example[0]
    targets = example[1]

    prediction = model.predict(inputs)

    actual_class = targets['symbol_rate'].numpy()[0][0]
    actual_symbol_rate = inverse_symbol_rate_dict[actual_class]

    best_class = np.argmax(prediction, axis=-1)[0]
    best_symbol_rate = inverse_symbol_rate_dict[best_class]

    i = inputs['I'].numpy().squeeze()
    q = inputs['Q'].numpy().squeeze()

    complex_input = i + 1j * q

    fft_exponentials = np.fft.fft(complex_input)
    fft_energies = np.abs(fft_exponentials)
    fft_spectrum = 20 * np.log10(np.fft.fftshift(fft_energies))

    fft_bin_size_kHz = (25600 / 1024) / 1000

    frequencies_kHz = np.arange(-12.8, 12.8, fft_bin_size_kHz)

    plt.plot(frequencies_kHz, fft_spectrum)
    plt.title(f"Power Spectrum, {actual_symbol_rate} sps, prediction: {best_symbol_rate} sps")
    plt.ylabel("Power (dB)")
    plt.xlabel("Frequency (kHz)")
    plt.xticks(np.arange(-13, 13, 1))

    print(f"{index}: Expected: {actual_symbol_rate}, Predicted: {best_symbol_rate}")

    plt.show()
