import tensorflow as tf
from tensorflow.data import AUTOTUNE
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from preprocess import resize_width, resize_height

data_dir = './dataset/'
val_dir = './validation/'
batch_size = 8

# train_ds = tf.keras.utils.image_dataset_from_directory(
#     data_dir,
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     image_size=(resize_height, resize_width),
#     batch_size=batch_size)

# val_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   image_size=(resize_height, resize_width),
#   batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE

datagen = ImageDataGenerator(horizontal_flip=True, zoom_range=(1, 1.5))

train_generator = datagen.flow_from_directory(
        data_dir, 
        target_size=(resize_width, resize_height), 
        batch_size=batch_size,
        subset="training",
        class_mode='binary')

datagen2 = ImageDataGenerator()
val_generator = datagen2.flow_from_directory(
        val_dir, 
        target_size=(resize_width, resize_height), 
        batch_size=batch_size,
        subset="validation",
        class_mode='binary',
        shuffle = False)

# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(resize_height, resize_width, 3)),
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(128, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=RMSprop(lr=0.001),
    loss='binary_crossentropy',
    metrics=['acc'])

checkpoint_filepath = './tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)

callbacks = [
    EarlyStopping(patience=2),
    model_checkpoint_callback,
]

# history = model.fit(train_ds, epochs=600,
#                     validation_data=val_ds,
#                     callbacks=callbacks)

history = model.fit(train_generator, epochs=600,
                    callbacks=callbacks,
                    validation_data=val_generator)

model.save("model.h5")

metrics_df = pd.DataFrame(history.history)
metrics_df[["loss","val_loss", "acc", "val_acc"]].plot()
plt.show()