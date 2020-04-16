import cv2
from PIL import Image
import numpy as np

# Read/Write Images with OpenCV
image = cv2.imread("lena.jpg")
cv2.imwrite("out1.png", image)

# Read/Write Images with PIL
image = Image.open("lena.jpg")
image.save("out2.png")
# 1
# 2
# Read/Write Images with PyOpenCV
# 
# mat = pyopencv.imread(“ponzo.jpg”)
# pyopencv.imwrite(“out.png”, mat)
# 1
# 2
# Convert between OpenCV image and PIL image
# 
# # color image
# cimg = cv.LoadImage("ponzo.jpg", cv.CV_LOAD_IMAGE_COLOR)     # cimg is a OpenCV image
# pimg = Image.fromstring("RGB", cv.GetSize(cimg), cimg.tostring())    # pimg is a PIL image
# cimg2 = cv.CreateImageHeader(pimg.size, cv.IPL_DEPTH_8U, 3)      # cimg2 is a OpenCV image
# cv.SetData(cimg2, pimg.tostring())
# 
# Note: OpenCV stores color image in BGR format. So, the converted PIL image is also in BGR-format. The standard PIL image is stored in RGB format.
# 
# # gray image
# cimg = cv.LoadImage("ponzo.jpg", cv.CV_LOAD_IMAGE_GRAYSCALE)   # cimg is a OpenCV image
# pimg = Image.fromstring("L", cv.GetSize(cimg), cimg.tostring())                 # img is a PIL image
# cimg2 = cv.CreateImageHeader(pimg.size, cv.IPL_DEPTH_8U, 1)              # img2 is a OpenCV image
# cv.SetData(cimg2, pimg.tostring())
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 10
# 11
# 12
# 13
# Convert between PIL image and NumPy ndarray
# 
# image = Image.open(“ponzo.jpg”)   # image is a PIL image
# array = numpy.array(image)          # array is a numpy array
# image2 = Image.fromarray(array)   # image2 is a PIL image
# 1
# 2
# 3
# Convert between PIL image and PyOpenCV matrix
# 
# image = Image.open(“ponzo.jpg”)                  # image is a PIL image
# mat = pyopencv.Mat.from_pil_image(image)  # mat is a PyOpenCV matrix
# image2 = mat.to_pil_image()                        # image2 is a PIL image
# 1
# 2
# 3

# Convert between OpenCV image and NumPy ndarray
cimg = cv2.imread("lena.jpg", cv2.IMREAD_COLOR)  # cimg is a OpenCV image
pimg = Image.frombytes("RGB", (cimg.shape[0], cimg.shape[1]), cimg.tostring())  # pimg is a PIL image
array = np.array(pimg)  # array is a numpy array， 这个array就是image
cv2.imshow('1-1', array)
cv2.waitKey(0)

#pimg2 = cv2.fromarray(array)  # pimg2 is a OpenCV image'

'''
后来网上搜索后才知道，对于opencv的像素是BGR顺序，然而matplotlib所遵循的是RGB顺序。
opencv的一个像素为：[B,G,R] ,matplotlib的一个像素为：[R,G,B]。这就是为什么本来发红的区域变得有些发蓝了。
'''
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg', cv2.IMREAD_COLOR)
plt.imshow(img)
plt.show()
img = cv2.imread('lena.jpg', cv2.IMREAD_COLOR)

# method1
b, g, r = cv2.split(img)
img2 = cv2.merge([r, g, b])
plt.imshow(img2)
plt.show()

# method2
'''
二、说明：img[:,:,::-1]
中括号中有两个逗号，四个冒号
[:, :, ::-1]
第一个冒号——取遍图像的所有行数
第二个冒号——取遍图像的所有列数
第三个和第四个冒号——取遍图像的所有通道数，-1是反向取值
'''
img3 = img[:, :, ::-1]
plt.imshow(img3)
plt.show()

# method3
img4 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img4)
plt.show()

#cv2.imshow("cv-np", pimg2)
# list转numpy
# 45 img = np.asarray(img_list)
# 46 # 还原图片
# 47 cv2.imwrite(IMAGE_NAME,img)

# PIL.Image转换成OpenCV格式：
image = Image.open("lena.jpg")
image.show()
img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
cv2.imshow("OpenCV", img)
cv2.waitKey()

# OpenCV转换成PIL.Image格式：
img = cv2.imread("lena.jpg")
cv2.imshow("OpenCV", img)
image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
image.show()
cv2.waitKey()
