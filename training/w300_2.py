import re
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 注意文件目录
file_path_indoor = r"G:/dataset/300w/300W/01_Indoor/"
file_path_outdoor = r"G:/dataset/300w/300W/02_Outdoor/"
# os.listdir() 可以获得指定的文件夹包含的文件或文件夹的名字的列表
indoor_file = os.listdir(file_path_indoor)
outdoor_file = os.listdir(file_path_outdoor)
print(indoor_file[0:10], "\n", outdoor_file[0:10])   # 打印 "G:/300W/01_Indoor/" 以及 "G:/300W/02_Outdoor/" 各 10 个文件

image_size_h, image_size_w = 224, 224                                      # 用来保存图片的尺寸
files_num = int(len(indoor_file)/2) + int(len(outdoor_file)/2)            # 用来保存所有的图片数量
facial_keypoints = np.zeros([files_num, 68*2], dtype=np.float64)          # 用来保存人脸关键点数据
facial_image = np.zeros([files_num, image_size_h, image_size_w, 3], dtype=np.uint8)            # 用来保存人脸图片数据
image_scale = np.zeros([files_num, 2], dtype=np.float64)                  # 用来保存图片的缩放比

count = 0  # 用来计数

# 读取 "G:/300W/01_Indoor/" 里面的图片和人脸关键点数据
for i in indoor_file:
    ret = re.match(r"(.\w+)\.pts", i)  # 使用正则表达式读取 .pts 文件
    if ret:
        # 读取 pts 点
        pts_file = file_path_indoor + i
        with open(pts_file, "r") as f:
            pts_str = f.read()
        # 读取当中的关键点数据
        key_point = re.findall(r"\d+\.\d+", pts_str)
        # 将列表中的字符串转化为数字
        key_point = [float(x) for x in key_point]
        # 添加到 facial_keypoints
        facial_keypoints[count] = key_point

        # 读取图片文件
        image_file = file_path_indoor + ret.group(1) + ".png"
        # 打开图片
        image = cv.imread(image_file, cv.IMREAD_COLOR)
        # 获取图片长和宽
        image_h, image_w = image.shape[:2]
        # 改变图片尺寸
        image = cv.resize(image, (image_size_h, image_size_w))
        # 得到缩放比
        image_scale[count] = [image_h / image_size_h, image_w / image_size_w]
        # 添加到 facial_image
        facial_image[count] = image

        count += 1

# 读取 "G:/300W/01_Outdoor/" 里面的图片和人脸关键点数据
for i in outdoor_file:
    ret = re.match(r"(.\w+)\.pts", i)
    if ret:
        # 读取 pts 点
        pts_file = file_path_outdoor + i
        with open(pts_file, "r") as f:
            pts_str = f.read()
        # 读取当中的关键点数据
        key_point = re.findall(r"\d+\.\d+", pts_str)
        # 将列表中的字符串转化为数字
        key_point = [float(x) for x in key_point]
        # 添加到 facial_keypoints
        facial_keypoints[count] = key_point

        # 读取图片文件
        image_file = file_path_outdoor + ret.group(1) + ".png"
        # 打开图片
        image = cv.imread(image_file, cv.IMREAD_COLOR)
        # 获取图片长和宽
        image_h, image_w = image.shape[:2]
        # 改变图片尺寸
        image = cv.resize(image, (image_size_h, image_size_w))
        # 得到缩放比
        image_scale[count] = [image_h / image_size_h, image_w / image_size_w]
        # 添加到 facial_image
        facial_image[count] = image

        count += 1

count = 0  # 用来计数

for i in image_scale:
    facial_keypoint = facial_keypoints[count]        # 获取单个人脸关键点数据
    facial_keypoint[::2] = facial_keypoint[::2] / i[1] / image_size_w # 对人脸关键点数据的宽进行缩放
    facial_keypoint[1::2] = facial_keypoint[1::2] / i[0] / image_size_h # 对人脸关键点数据的长进行缩放
    count += 1

for j in range(10):    # 总共展示十张图片
    image = facial_image[j]    # 获取图片数据
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)   # 由于 matplotlib 支持的是 rgb 色彩空间，所以我们需要将 bgr 转化为 rgb
    for i in range(0, 68):
        if i in [27,28,29,32,33]:
            cv.circle(image, (int(facial_keypoints[j, 2*i] * image_size_w), int(facial_keypoints[j, 2*i+1] * image_size_h)), 5, (255, 0, 0), -1)   # 进行打点
    plt.subplot(2, 5, j+1)
    plt.imshow(image)
plt.show()

np.save("./facial_image.npy", facial_image)
np.save("./facial_keypoints.npy", facial_keypoints)

facial_image = np.load(r"./facial_image.npy")
facial_keypoints = np.load(r"./facial_keypoints.npy")
