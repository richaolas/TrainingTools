import tensorflow as tf

import numpy as np

class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,             # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            activation=tf.nn.relu   # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)                  # [batch_size, 28, 28, 32]
        x = self.pool1(x)                       # [batch_size, 14, 14, 32]
        x = self.conv2(x)                       # [batch_size, 14, 14, 64]
        x = self.pool2(x)                       # [batch_size, 7, 7, 64]
        x = self.flatten(x)                     # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)                      # [batch_size, 1024]
        x = self.dense2(x)                      # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)  # [60000]
        self.test_label = self.test_label.astype(np.int32)  # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]

#
# class MLP(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.flatten = tf.keras.layers.Flatten()    # Flatten层将除第一维（batch_size）以外的维度展平
#         self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
#         self.dense2 = tf.keras.layers.Dense(units=10)
#
#     def call(self, inputs):         # [batch_size, 28, 28, 1]
#         x = self.flatten(inputs)    # [batch_size, 784]
#         x = self.dense1(x)          # [batch_size, 100]
#         x = self.dense2(x)          # [batch_size, 10]
#         output = tf.nn.softmax(x)
#         return output
#
num_epochs = 5
batch_size = 50
learning_rate = 0.001

model = CNN()
data_loader = MNISTLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

X, y = data_loader.get_batch(batch_size)
print(X.shape)
model.build(input_shape=(None,28,28,1))
#print(model.variables)

print(model.summary())

for v in model.variables:
    print(v.name, v.shape)



num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

#print(model.variables)

# X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
# y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)
#
# X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
# y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
#
# print(X, y)
#
# X = tf.constant(X)
# y = tf.constant(y)
#
# X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# y = tf.constant([[10.0], [20.0]])
#
#
# class Linear(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.dense = tf.keras.layers.Dense(
#             units=1,
#             activation=None,
#             kernel_initializer=tf.zeros_initializer(),
#             bias_initializer=tf.zeros_initializer()
#         )
#
#     def call(self, input):
#         output = self.dense(input)
#         return output
#
# model = Linear()
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
#
# for i in range(100):
#     with tf.GradientTape() as tape:
#         y_pred = model(X)
#         loss = tf.reduce_mean(tf.square(y - y_pred))
#     grads = tape.gradient(loss, model.variables)
#     optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
#
# print(model.variables)
#


# a = tf.Variable(initial_value=0.)
# b = tf.Variable(initial_value=0.)
#
# num_epoch = 1000
# '''
#  """Stochastic gradient descent and momentum optimizer.
# '''
# optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
#
# for e in range(num_epoch):
#     #print(e)
#     with tf.GradientTape() as tape:
#         y_pred = a * X + b
#         loss = tf.reduce_sum(tf.square(y_pred - y))
#     grads = tape.gradient(loss, [a, b])
#     optimizer.apply_gradients(grads_and_vars=zip(grads, [a, b]))
#
# print(a, b)


# # 定义一个随机数（标量）
# random_float = tf.random.uniform(shape=())
#
# # 定义一个有2个元素的零向量
# zero_vector = tf.zeros(shape=(2))
#
# # 定义两个2×2的常量矩阵
# A = tf.constant([[1., 2.], [3., 4.]])
# B = tf.constant([[5., 6.], [7., 8.]])
#
# print(random_float, zero_vector, A, B)
