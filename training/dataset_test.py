import os

import numpy as np
import tensorflow as tf

# def count(stop):
#   i = 0
#   while i<stop:
#     print('第%s次调用……'%i)
#     yield i
#     i += 1
#
# dataset2 = tf.data.Dataset.from_generator(count, args=[3], output_types=tf.int32, output_shapes = (), )
# a = iter(dataset2)
# next(a)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input_path = glob.glob('D:/PycharmProjects/testSave/TrainData/input/*.jpg')  # 图片的路径
# label_path = glob.glob('D:/PycharmProjects/testSave/TrainData/label/*.jpg')  # 图片的路径
#
#
# def load_preprosess_image(input_path, label_path):
#     image = tf.io.read_file(input_path)  # 读取的是二进制格式 需要进行解码
#     image = tf.image.decode_jpeg(image, channels=3)  # 解码 是通道数为3
#     image = tf.image.resize(image, [256, 256])  # 统一图片大小
#     image = tf.cast(image, tf.float32)  # 转换类型
#     image = image / 255  # 归一化
#
#     label = tf.io.read_file(label_path)
#     label = tf.image.decode_jpeg(label, channels=3)  # 解码 是通道数为3
#     label = tf.image.resize(label, [256, 256])  # 统一图片大小
#     label = tf.cast(label, tf.float32)  # 转换类型
#     label = label / 255  # 归一化
#
#     return image, label  # return回的都是一个batch一个batch的 ， 一个批次很多张
#
#
# train_dataset = tf.data.Dataset.from_tensor_slices((input_path, label_path))  # 用load_preprosess_image对图片做一个读取预处理 速度有些慢
#
# AUTOTUNE = tf.data.experimental.AUTOTUNE  # 根据计算机cpu的个数自动的做并行运算  临时实验方法 有可能变化
#
# train_dataset = train_dataset.map(load_preprosess_image,
#                                   num_parallel_calls=AUTOTUNE)  # .map是使函数应用在load_preprosess_image中所有的图像上
#
# BATCH_SIZE = 2
# train_count = len(input_path)
#
# train_dataset = train_dataset.shuffle(train_count).batch(BATCH_SIZE)
#
# train_dataset = train_dataset.prefetch(AUTOTUNE)  # 前台在训练时 后台读取数据 自动分配cpu
#
# imgs, las = next(iter(train_dataset))  # 取出的是一个batch个数的图片 shape = (batch_size,256,256,3)
#
# plt.imshow(imgs[1])  # 展示我们读到的图像
# plt.show()

facial_image = np.load(r"./facial_image.npy")
facial_keypoints = np.load(r"./facial_keypoints.npy")

facial_image = facial_image.astype(np.float32)
facial_image = facial_image / 255.0

# facial_keypoints = facial_keypoints[:,29*2:30*2]
facial_keypoints = facial_keypoints[:,
                   [27 * 2, 27 * 2 + 1, 28 * 2, 28 * 2 + 1, 29 * 2, 29 * 2 + 1, 32 * 2, 32 * 2 + 1, 33 * 2, 33 * 2 + 1]]

np.save("./facial_image3.npy", facial_image)
np.save("./facial_keypoints3.npy", facial_keypoints)

# print(facial_image, facial_keypoints)
print(facial_image.shape, facial_keypoints.shape)
image_dataset = tf.data.Dataset.from_tensor_slices(facial_image)
keypoint_dataset = tf.data.Dataset.from_tensor_slices(facial_keypoints)

image_keypoint_ds = tf.data.Dataset.zip((image_dataset, keypoint_dataset))  # 图片和标签整合


def get_one(dataset):
    for elment in dataset:
        return elment


s = get_one(image_keypoint_ds)  # (500,500,3), (136)
print(s)
