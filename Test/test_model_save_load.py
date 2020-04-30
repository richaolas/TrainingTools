from tensorflow.keras import layers, Sequential
import tensorflow as tf

network = Sequential([ # 封装为一个网络
layers.Dense(3, activation=None), # 全连接层
#layers.ReLU(),#激活函数层
#layers.Dense(2, activation=None), # 全连接层
#layers.ReLU() #激活函数层
])
x = tf.random.normal([4,3])
network.build(input_shape=(4,3))
y = network.predict(x)
#y = network(x) # 输入从第一层开始，逐层传播至最末层

# tf.saved_model.save(network, export_dir = "test")
# network2 = tf.saved_model.load("test")
# print(network2)
# y2 = network2(x)
# print(y.numpy(), y2.numpy())

#network.save('model.h5')
#print('saved total model.')

#tf.saved_model.save(network, 'model-savedmodel')
#tf.keras.experimental.export_saved_model(network, 'model-savedmodel')
#tf.keras.models.save_model(network, 'model-savedmodel')
#print('export saved model.')
#print('saving savedmodel.')
tf.saved_model.save(network, 'tmp_saved_model')
#tf.keras.experimental.export_saved_model(network, 'saved_model_path')
#network.save('saved_model_path', save_format="tf")
#tf.keras.models.save_model(network, 'saved_model_path', save_format="tf")
network.summary()
del network # 删除网络对象

#x = tf.random.normal([4,3])
#
# print('loaded model from file.')
# #network2 = tf.keras.models.load_model('model.h5', compile=False)
# #network2 = tf.keras.models.load_model('model-savedmodel')
imported = tf.saved_model.load('tmp_saved_model')
#loaded = tf.saved_model.load("/tmp/mobilenet/1/")
print(list(imported.signatures.keys()))  # ["serving_default"]
# network2= imported.signatures["serving_default"]
# print(network2)
# #network2 = tf.keras.experimental.load_from_saved_model('model-savedmodel')
# #network2.build(input_shape=(4,3))
# y2 = network2(x)
#
#
# print(y.numpy(), y2.numpy())