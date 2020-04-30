from tensorflow.keras import layers, Sequential
import tensorflow as tf

print('loaded model from file.')
network = tf.keras.models.load_model('model.h5', compile=False)

x = tf.random.normal([4,3])
y2 = network(x)
print(y2)
