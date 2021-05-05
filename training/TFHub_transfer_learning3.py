# geektutu.com
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, datasets
import sys
import os
import cv2
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

imported = tf.saved_model.load("saved_model_keypoint12") #直接Load

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    frame = cv2.imread("indoor_010.png")
    print(frame.shape)
    h, w, _ = frame.shape
    img = cv2.resize(frame, (224, 224))
    img = np.array(img).reshape(1, 224, 224, 3) / 255.0
    img = img.astype(np.float32)
    result = imported(img)

    for i in range(0,len(result[0]),2):
        x = w * result[0][i]
        y = h * result[0][i+1]
        ret = cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)
    cv2.imshow("show", ret)
    cv2.waitKey(1)

image_file = "indoor_010.png"
tutu = Image.open(image_file).resize((224, 224))
#tutu.show()
img = np.array(tutu).reshape(1, 224, 224, 3)/255.0
img = img.astype(np.float32)
print(img, img.shape, img.dtype)
# geektutu.com
result = imported(img)

#result = model.predict(tf.expand_dims(s1[0], axis=0))
print(result)


#[0.4449998 0.516709 ]
# 0.50369227 0.4235118

image = cv2.imread(image_file)
h, w, c = image.shape
x = w * result[0][0]
y = h * result[0][1]

ret = cv2.circle(image, (int(x),int(y)), 5, (255, 0, 0), -1)
cv2.imshow("show", ret)
cv2.waitKey(0)


sys.exit()











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


