import matplotlib.pyplot as plt
from math import *
from numpy import *
import numpy as np
#plt.figure(figsize=(12,6))




# ax1 = plt.subplot(231)
# ax1.axis([-1, 1, -1, 1])
# plt.subplot(232)
# plt.subplot(233)
# plt.subplot(234)
# plt.subplot(235)
# plt.subplot(236)

x = [1, 2, 3, 4, 5]
y = [2.3, 3.4, 1.2, 6.6, 7.0]
#z = [1,2,3,4,5]
#A.画散点图*
#plt.scatter(x, y, color='#FF0000', marker='+')
#plt.show()



fig = plt.figure(figsize=(12, 6))
plt.subplot(331)
plt.plot(x, y, color='r', linestyle='-')
plt.subplot(332)
plt.plot(x, y, color='r', linestyle='--')
plt.subplot(333)
x1 = np.arange(-math.pi, math.pi, 0.01)
y1 = np.sin(x1) #[sin(xx) for xx in x1]
X,Y = np.meshgrid(np.arange(-math.pi, math.pi, 0.5), np.arange(-math.pi, math.pi, 0.5))
plt.plot(x1, y1, color='r', linestyle='-.')
plt.scatter([1,2,3,4],[[1,2],[3,4]])

y = [2.3, 3.4, 1.2, 6.6, 7.0]
plt.subplot(334)
#plt.figure()
plt.pie(y)
plt.title('PIE', y=-0.1)

plt.subplot(335)
plt.scatter(x, y, color='#FF0000', marker='+')

x = [1, 2, 3, 4, 5]
y = [2.3, 3.4, 1.2, 6.6, 7.0]

plt.subplot(336)
plt.bar(x, y)
plt.title("bar", y=-0.3)

#####################
delta = 0.025
x = y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z = Y**2 + X**2
#plt.figure(figsize=(12, 6))
plt.subplot(337)
plt.contour(X, Y, Z)
plt.colorbar()
plt.title("contour")


plt.show()


#plt.show()

"""
这里的参数意义：

x为横坐标向量，y为纵坐标向量，x,y的长度必须一致。
控制颜色：color为散点的颜色标志，常用color的表示如下：

b---blue   c---cyan  g---green    k----black
m---magenta r---red  w---white    y----yellow
有四种表示颜色的方式:

用全名
16进制，如：#FF00FF
灰度强度，如：‘0.7’
控制标记风格：marker为散点的标记，标记风格有多种：

.  Point marker
,  Pixel marker
o  Circle marker
v  Triangle down marker 
^  Triangle up marker 
<  Triangle left marker 
>  Triangle right marker 
1  Tripod down marker
2  Tripod up marker
3  Tripod left marker
4  Tripod right marker
s  Square marker
p  Pentagon marker
*  Star marker
h  Hexagon marker
H  Rotated hexagon D Diamond marker
d  Thin diamond marker
| Vertical line (vlinesymbol) marker
_  Horizontal line (hline symbol) marker
+  Plus marker
x  Cross (x) marker
"""

"""
这里有一个新的参数linestyle，控制的是线型的格式：符号和线型之间的对应关系

-      实线
--     短线
-.     短点相间线
：     虚点线
"""