import keras
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from dataset_management import full_dataset

model = tf.keras.models.load_model("model.keras")

model.summary()

n_predictions = 64

predict_dataset = full_dataset().shuffle(buffer_size=1024).take(n_predictions)

predictions = model.predict(predict_dataset.batch(1))

for index, example in predict_dataset.enumerate():
    prediction = predictions[index]

    inputs = example[0]
    targets = example[1]

    actual_symbol_rate = targets['symbol_rate']
    best_symbol_rate = np.argmax(prediction)

    i = inputs['I'].numpy()
    q = inputs['Q'].numpy()

    complex_input = i + 1j * q

    #fft_exponentials = np.fft.fft(complex_input)
    #fft_energies = np.abs(fft_exponentials)
    #fft_spectrum = np.fft.fftshift(fft_energies)

    plt.plot(np.unwrap(np.angle(complex_input)))

    print(f"{index}: Expected: {actual_symbol_rate}, Predicted: {best_symbol_rate}")

    plt.show()

    input("blah")
