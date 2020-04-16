import tensorflow as tf
import numpy as np

#np.save('data.npy',np.ones(1024))


# def func(mystr):
#     return np.load(mystr.numpy())
#
# mystring = tf.constant('data.npy')
# data = tf.data.Dataset.from_tensor_slices([mystring])
# data.map(func,1)
d = tf.data.Dataset.from_tensor_slices(['hello', 'world'])

#  transform a byte string tensor to a byte numpy string and decode to python str
#  upper case string using a Python function
def upper_case_fn(t):
    return t.numpy().decode('utf-8').upper()

#  use the python code in graph mode
d.map(lambda x: tf.py_function(func=upper_case_fn,
      inp=[x], Tout=tf.string))  # ==> [ "HELLO", "WORLD" ]