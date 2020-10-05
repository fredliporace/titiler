from keras.models import load_model
import tensorflow as tf

NETWORK_FILE = "./keras-models/cloud_segmentation_20200923_1844.h5"
MODEL = load_model(NETWORK_FILE)

converter = tf.lite.TFLiteConverter.from_keras_model(MODEL)
tflite_mode = converter.convert()
OUTPUT_FILE = NETWORK_FILE.replace('.h5','.tflite')
with open(OUTPUT_FILE, 'wb') as f:
    f.write(tflite_mode)
