#!/bin/env python

import tensorflow as tf
from keras.models import load_model

NETWORK_FILE = "./keras-models/cloud_segmentation_20201006_1917.h5"
MODEL = load_model(NETWORK_FILE)

converter = tf.lite.TFLiteConverter.from_keras_model(MODEL)
tflite_mode = converter.convert()
OUTPUT_FILE = NETWORK_FILE.replace(".h5", ".tflite")
with open(OUTPUT_FILE, "wb") as f:
    f.write(tflite_mode)
