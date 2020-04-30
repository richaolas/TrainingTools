import tensorflow as tf
import os
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, datasets

network = tf.keras.models.load_model('model.h5', custom_objects={"KerasLayer": hub.KerasLayer})
#tf.keras.experimental.export_saved_model(network, 'model-savedmodel')
tf.keras.models.save_model(network, filepath='savedmodel', save_format="tf")
tf.saved_model.save(network, export_dir='savedmodel2')
path = r'F:\data\2020-04-09'

for file in os.listdir(path):
    if os.path.isdir(os.path.join(path, file)):
        continue
    print(os.path.join(path, file))
    image = cv2.imread(os.path.join(path, file))
    show_image = image[:,:,:]
    #  (320, 80, 320+300, 80 + 200)
    image = image[80:80+200, 320:320+300,  :]
    #show_image = image[:, :, :]
    image = cv2.resize(image, (224, 224))
    image = image / 255.0 - 0.5
    image = tf.expand_dims(image, axis=0)
    ret = network.predict(image)
    print(tf.argmax(ret, axis=1))
    result = tf.argmax(ret, axis=1).numpy()
    if result == 1:
        cv2.rectangle(show_image, (320, 80), (320+300,80+200), (0,0,255), 2)
    cv2.imshow('', show_image)
    cv2.waitKey(0)
