import tensorflow as tf

from dataset_management import create_datasets
from model_definition import compiled_model
from tensorboard_config import log_dir

train_dataset, validation_dataset, test_dataset = create_datasets()

model = compiled_model()

tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    write_images=True,
    histogram_freq=0,
    embeddings_freq=0,
    update_freq="epoch"
)

learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=0.0001)

stop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, verbose=1)

callbacks = [tb_callback, learning_rate_callback, stop_callback]

model.fit(train_dataset, epochs=100, validation_data=validation_dataset, callbacks=callbacks)

model.evaluate(test_dataset)

model.save("model.keras")
