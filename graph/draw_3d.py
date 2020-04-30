"""
1.创建三维坐标轴对象Axes3D
创建Axes3D主要有两种方式，一种是利用关键字 projection='3d' 来实现，
另一种则是通过从mpl_toolkits.mplot3d导入对象Axes3D来实现，目的都是生成具有三维格式的对象Axes3D.
"""

import numpy as np
import math

# 方法一，利用关键字
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义坐标轴
# fig = plt.figure()
x = [1, 2, 3, 4, 5]
y = [2.3, 3.4, 1.2, 6.6, 7.0]
# z = [1,2,3,4,5]
# A.画散点图*
# plt.scatter(x, y, color='#FF0000', marker='+')
# plt.show()

fig = plt.figure(figsize=(12, 6))

# 三维曲面
# 定义三维数据
xx = np.arange(-5, 5, 0.5)
yy = np.arange(-5, 5, 0.5)
X, Y = np.meshgrid(xx, yy)
Z = np.sin(X) + np.cos(Y)

# 作图
ax1 = fig.add_subplot(221, projection='3d')  # 这种方法也可以画多个子图
ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
ax1.contour(X, Y, Z, zdim='z', offset=-2, cmap='rainbow')  # 等高线图，要设置offset，为Z的最小值

# 随机散点图
ax2 = fig.add_subplot(222, projection='3d')  # 这种方法也可以画多个子图

#生成三维数据
xx = np.random.random(20)*10-5   #取100个随机数，范围在5~5之间
yy = np.random.random(20)*10-5
X, Y = np.meshgrid(xx, yy)
Z = np.sin(np.sqrt(X**2+Y**2))

#作图
ax2.scatter(X,Y,Z,alpha=0.3,c=np.random.random(400),s=np.random.randint(10,20, size=(20, 40)))     #生成散点.利用c控制颜色序列,s控制大小

#设定显示范围


# 三维曲线和散点
ax3 = fig.add_subplot(223, projection='3d')  # 这种方法也可以画多个子图
z = np.linspace(0, 13, 1000)
x = 5 * np.sin(z)
y = 5 * np.cos(z)
zd = 13 * np.random.random(100)
xd = 5 * np.sin(zd)
yd = 5 * np.cos(zd)
ax3.scatter3D(xd, yd, zd, cmap='Blues')  # 绘制散点图
ax3.plot3D(x, y, z, 'gray')  # 绘制空间曲线
# plt.show()

#
ax4 = fig.add_subplot(224, projection='3d')  # 这种方法也可以画多个子图
z = np.linspace(0, 13, 1000)
r = 10
theta = np.linspace(0,2*math.pi,100)
x = r * np.cos(theta)
y = r * np.sin(theta)
z = [1]*100
ax4.scatter3D(x, y, z, cmap='Blues')

# plt.subplot(333)
# ax2 = Axes3D(fig) # 子图不可以

# plt.axes(projection='3d')
# #ax = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图
# plt.subplot(333)
# plt.axes(projection='3d')

# 方法二，利用三维轴方法
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# #定义图像和三维格式坐标轴
# fig=plt.figure()
# ax2 = Axes3D(fig)
plt.show()
