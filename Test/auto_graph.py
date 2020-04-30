import tensorflow as tf
import numpy as np

@tf.function(autograph=True)
def myadd(a,b):
    for i in tf.range(3):
        tf.print(i)
    c = a+b
    tf.print("tracing")
    return c

myadd(tf.constant("hello"),tf.constant("world"))

myadd(tf.constant("hellox"),tf.constant("worldg"))


class DemoModule(tf.Module):
    def __init__(self, init_value=tf.constant(0.0), name=None):
        super(DemoModule, self).__init__(name=name)
        with self.name_scope:  # 相当于with tf.name_scope("demo_module")
            self.x = tf.Variable(init_value, dtype=tf.float32, trainable=True) # 'demo_module/Variable:0'
        self.y = tf.Variable(init_value, dtype=tf.float32, trainable=True)     # 'Variable:0'

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
    def addprint(self, a):
        with self.name_scope:
            self.x.assign_add(a)
            tf.print(self.x)
            return (self.x)

#执行
demo = DemoModule(init_value = tf.constant(1.0))
result = demo.addprint(tf.constant(5.0))

#查看模块中的全部变量和全部可训练变量
print(demo.variables)
print(demo.trainable_variables)

import tensorflow as tf
from tensorflow.keras import models,layers,losses,metrics

model = models.Sequential()

model.add(layers.Dense(4,input_shape = (10,)))
model.add(layers.Dense(2))
model.add(layers.Dense(1))
model.summary()

print(model.submodules)
print(model.layers)

'''
Keras 是高层接口: tf.keras

对于常见的神经网络层：
1. 可以使用张量方式的底层接口函数实现  tf.nn
2. 一般直接使用层 tf.keras.layers，只需要在创建时指定网络层的相关参数，并调用__call__方法即可完成前向计算

在 Keras 中，有2 个比较特殊的类：keras.Model 和keras.layers.Layer 类。其中Layer
类是网络层的母类，定义了网络层的一些常见功能，如添加权值，管理权值列表等。
Model 类是网络的母类

通过 Model.predict(x)方法即可完成模型的预测


class MyModel(keras.Model):



'''