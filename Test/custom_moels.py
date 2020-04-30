import tensorflow as tf


class CustomModule(tf.keras.Model):

    def __init__(self):
        super(CustomModule, self).__init__()
        self.v = tf.Variable(1.)

    #@tf.function
    #def __call__(self, x):
    #    return x * self.v
    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
    def call(self, inputs, training=None, mask=None):
        return inputs * self.v

    # @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
    def mutate(self, new_v):
        self.v.assign(new_v)


# x = tf.constant(5.0)
module = CustomModule()
module.build(input_shape=())
print(module.summary())

#call = module.__call__.get_concrete_function(x=tf.TensorSpec((), tf.float32))
#call = module.call.get_concrete_function(x=tf.TensorSpec((), tf.float32))
# print(module(x))

tf.saved_model.save(module, "module_no_signatures")
tf.saved_model.save(module, "module_with_signature", signatures = {"serving_default": module.call})
print('end saving')
#tf.saved_model.save(module, "module_with_signature", signatures=call)

x = tf.constant(5.0)
#imported = tf.saved_model.load("module_no_signatures")
imported = tf.saved_model.load("module_with_signature")
print(imported.call(x))
print(imported.signatures)
infer = imported.signatures['serving_default']
print(infer(x))


#
# # print(type(loaded), )
# # infer = loaded2.signatures['serving_default']
# # print(loaded(x), loaded2(x), infer(x)['output_0'])
#
# @tf.function(input_signature=[tf.TensorSpec([], tf.string)])
# def parse_string(string_input):
#     return imported(tf.strings.to_number(string_input))
#
#
# signatures = {"serving_default": parse_string,
#               "from_float": imported.signatures["serving_default"]}
#
# tf.saved_model.save(imported, "module_with_multiple_signatures", signatures)
#
# imported2 = tf.saved_model.load("module_with_multiple_signatures")
# print(imported2.signatures)
#
# infer1 = imported2.signatures['serving_default']
# infer2 = imported2.signatures['from_float']
#
# print(imported2(x), infer2(x), infer1(tf.constant('3.0')))
#print(infer2(tf.constant('3.0')))