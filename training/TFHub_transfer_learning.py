# geektutu.com
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, datasets

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

import tfrecord_to_dataset

def cn_mirror_net(url):
    return url.replace('tfhub.dev', 'hub.tensorflow.google.cn')

url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
url = cn_mirror_net(url)
print(url)
model = tf.keras.Sequential([
    hub.KerasLayer(url, input_shape=(224, 224, 3))
])

# geektutu.com
tutu = tf.keras.utils.get_file('tutu.png','https://geektutu.com/img/icon.png')
tutu = Image.open(tutu).resize((224, 224))
#tutu.show()

# geektutu.com
result = model.predict(np.array(tutu).reshape(1, 224, 224, 3)/255.0)
ans = np.argmax(result[0], axis=-1)
print('result.shape:', result.shape, 'ans:', ans)
# result.shape: (1, 1001) ans: 332

# geektutu.com
labels_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', labels_url)
imagenet_labels = np.array(open(labels_path).read().splitlines())
print(imagenet_labels[ans])
# hare

def resize(d, size=(224, 224)):
    return np.array([np.array(Image.fromarray(v).resize(size, Image.ANTIALIAS))
                     for i, v in enumerate(d)])

#tf..data.Dataset.

# (train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()
# train_x, test_x = resize(train_x[:10000])/255.0, resize(test_x)/255.0
# train_y = train_y[:10000]


feature_extractor_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
feature_extractor_url = cn_mirror_net(feature_extractor_url)
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))

# 这一层的训练值保持不变
feature_extractor_layer.trainable = False

model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(2, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
model.summary()

#history = model.fit(train_x, train_y, epochs=1)

# def read_and_decode(example_string):
'''
这里的参数要和创建样本时候的格式一致，tfrecord_to_dataset.py - parse_tf_to_image_lable                                                                                                                                                                                                                                                                                                                                                                                                 
'''
def read_and_decode(image, label, boxes):
    '''
    从TFrecord格式文件中读取数据
    '''
    # feature_dict = tf.io.parse_single_example(example_string, feature_description)
    # image = tf.io.decode_png(feature_dict['image'])
    # label = tf.io.decode_png(feature_dict['label'])
    # image = tf.cast(image, dtype='float32') / 255.
    # label = tf.cast(label, dtype='float32') / 255.
    #print(type(image), image.numpy().shape)
    #print(type(label), label)
    #img = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    #print(img)
    #label = tf.cast(features['label'], tf.int32)


    #tf.print(tf.shape(image))
    #image = tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
    #image = tf.image.crop_and_resize(image, boxes, [0], (224, 224), method = tf.image.ResizeMethod.BILINEAR)
    #image = tf.image.resize(image, (224, 224), method = tf.image.ResizeMethod.BILINEAR)
    shape = tf.shape(image)
    image = tf.reshape(image, [1, shape[0], shape[1], shape[2]])
    image = tf.image.crop_and_resize(image, boxes, [0], (224, 224), method=tf.image.ResizeMethod.BILINEAR)
    image = tf.reshape(image, [224, 224, -1])
    #image = tf.image.resize(image, (224, 224), method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    #tf.print(label)

    return image, label

#[Note] 需要创建好的一个dataset
#C:/atrain.record
dataset = tfrecord_to_dataset.tfrecord_to_dataset(r'C:/atrain2.record') #(r'C:\airconditionpin.tfrecord')
dataset = dataset.repeat() # 重复数据集
dataset = dataset.map(read_and_decode) # 解析数据
dataset = dataset.shuffle(buffer_size = 10000) # 在缓冲区中随机打乱数据
batch   = dataset.batch(batch_size = 64) # 每10条数据为一个batch，生成一个新的Datasets


#history = model.fit(batch, epochs=25, steps_per_epoch=500)
history = model.fit(batch, epochs=25, steps_per_epoch=8)
# loss, acc = model.evaluate(batch)  #model.evaluate(test_x, test_y)
# print(acc)

# save model
model.save('model.h5')
tf.saved_model.save(model, 'saved_model') #可以给其余语言使用的

imported = tf.saved_model.load("saved_model") #直接Load

for idx, (image, label) in enumerate(batch):
   #ret = imported.predict(image)
   ret = imported(image)
   print(ret, label)

#关于分类的简单训练方法：
#1. 准备样本
#2. 生成数据
#3. 训练
#4. 验证


