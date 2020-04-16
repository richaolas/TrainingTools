import tensorflow as tf
import os
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, datasets

network = tf.keras.models.load_model('model.h5',custom_objects={"KerasLayer": hub.KerasLayer})

path = r'H:\Samples\bandage'

for file in os.listdir(path):
    if os.path.isdir(os.path.join(path, file)):
        continue
    print(os.path.join(path, file))
    image = cv2.imread(os.path.join(path, file))
    image = cv2.resize(image, (224,224))
    image = image / 255.0 - 0.5
    image = tf.expand_dims(image, axis=0)
    ret = network.predict(image)
    print(tf.argmax(ret, axis=1))