from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')
model.summary()

network = keras.Sequential([ # 封装为一个网络
layers.Dense(3, activation=None), # 全连接层
#layers.ReLU(),#激活函数层
#layers.Dense(2, activation=None), # 全连接层
#layers.ReLU() #激活函数层
])
x = tf.random.normal([4,3])
network.build(input_shape=(4,3))
y1 = network.predict(x)

# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# x_train = x_train.reshape(60000, 784).astype('float32') / 255
# x_test = x_test.reshape(10000, 784).astype('float32') / 255
#
# model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               optimizer=keras.optimizers.RMSprop())
# history = model.fit(x_train, y_train,
#                     batch_size=64,
#                     epochs=1)
# # Reset metrics before saving so that loaded model has same state,
# # since metric states are not preserved by Model.save_weights
# model.reset_metrics()
#
# predictions = model.predict(x_test)

# Export the model to a SavedModel
network.save('path_to_saved_model2', save_format='tf')

# Recreate the exact same model
new_model = keras.models.load_model('path_to_saved_model2')
y2 = new_model.predict(x)

print(y1, y2)
import numpy as np

# Check that the state is preserved
#new_predictions = new_model.predict(x_test)
#np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

# Check that the state is preserved#
#new_predictions = new_model.predict(x_test)
#np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

#print(new_predictions)

# Note that the optimizer state is preserved as well:
# you can resume training where you left off.