import tensorflow as tf
from tensorflow.data import AUTOTUNE
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from preprocess import resize_width, resize_height

data_dir = './dataset/'
batch_size = 64

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(resize_height, resize_width),
    batch_size=batch_size)

print(train_ds.class_names)