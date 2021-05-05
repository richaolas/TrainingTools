'''
这个文件用来测试训练模型的
'''

import tensorflow as tf
import os
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, datasets

network = tf.keras.models.load_model('model.h5', custom_objects={"KerasLayer": hub.KerasLayer})
network.summary()
# tf.keras.experimental.export_saved_model(network, 'model-savedmodel')
tf.keras.models.save_model(network, filepath='savedmodel', save_format="tf")
tf.saved_model.save(network, export_dir='savedmodel2')
path = r'C:\Users\ric_r\source\repos\AirCondition\ConsoleApplication1\train\neg'

x, y, w, h = 0, 0, 50, 100

for file in os.listdir(path):
    if os.path.isdir(os.path.join(path, file)):
        continue
    print(os.path.join(path, file))
    image = cv2.imread(os.path.join(path, file))
    show_image = image[:, :, :]
    #  (320, 80, 320+300, 80 + 200)
    #image = image[y:y + h, x:x + w, :]
    # show_image = image[:, :, :]
    image = cv2.resize(image, (224, 224))
    cv2.imshow('src', image)
    image = image / 255.0 - 0.5
    image = tf.expand_dims(image, axis=0)
    ret = network.predict(image)
    print('[result]: ', ret)
    print(tf.argmax(ret, axis=1))
    result = tf.argmax(ret, axis=1).numpy()

    if result == 0:
        cv2.rectangle(show_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    elif result == 1:
        cv2.rectangle(show_image, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv2.imshow('', show_image)
    cv2.waitKey(0)
