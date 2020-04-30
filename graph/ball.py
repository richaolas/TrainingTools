import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D

# mpl模块载入的时候加载配置信息存储在rcParams变量中，rc_Params_from_file（）函数从文件加载配置信息
mpl.rcParams['legend.fontsize'] = 20
# 指定默认字体
mpl.rcParams['font.sans-serif'] = ['FangSong']
# 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False

font = {
    'color': 'b',
    'style': 'oblique',
    'size': 20,
    'weight': 'bold'

}

# 参数为图片大小
fig = plt.figure(figsize=(16, 16))
# get current axes,且坐标轴是3d的
ax = fig.gca(projection='3d')

# 准备数据
r = 1
# 两个角的参数
a = np.linspace(0, np.pi, 10)
b = np.linspace(0, 2 * np.pi, 10)

x = []
y = []
z = []

theta = np.arange(0, 2 * np.pi, 0.5)
phi = np.arange(0, 2 * np.pi, 0.5)

t,p = np.meshgrid(theta, phi)

# for i, j in product(theta, phi):
#     x.append(r * np.sin(i) * np.cos(j))
#     y.append(r * np.sin(i) * np.sin(j))
#     z.append(r * np.cos(i))

#ret = product(t, p)

# x = r * np.sin(t) * np.cos(p)
# x = np.reshape(x, (-1,))
# y = r * np.sin(t) * np.sin(p)
# y = np.reshape(y, (-1,))
# z = r * np.cos(t)
# z = np.zeros_like(y)
# 构造点阵
for i in a:
    for j in b:
        x.append(r * np.sin(i) * np.cos(j))
        y.append(r * np.sin(i) * np.sin(j))
        z.append(r * np.cos(i))

ax.set_xlabel("X", fontdict=font)
ax.set_ylabel("Y", fontdict=font)
ax.set_zlabel("Z", fontdict=font)

# 设置标题； alpha 参数指透明度 transparent
ax.set_title("3d circle", alpha=0.5, fontdict=font)
# label 图例
ax.plot(x, y, z, label='点阵球')
# legend（图例）的位置可选： upper right/left/center, lower right/left/center, best 等等
#ax.plot_surface(np.array(x), np.array(y), np.array(z), rstride=1, cstride=1, cmap='rainbow')
ax.legend(loc='upper right')

plt.show()