import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

file = tf.keras.utils.get_file(
    "grace_hopper.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
plt.imshow(img)
plt.axis('off')
x = tf.keras.preprocessing.image.img_to_array(img)
x = tf.keras.applications.mobilenet.preprocess_input(
    x[tf.newaxis,...])

labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

print(imagenet_labels)

pretrained_model = tf.keras.applications.MobileNet()

result_before_save = pretrained_model(x)

decoded = imagenet_labels[np.argsort(result_before_save)[0,::-1][:5]+1]

print("Result before saving:\n", decoded)
#
# tf.saved_model.save(pretrained_model, "./mobilenet/1/")
#
# loaded = tf.saved_model.load("./mobilenet/1/")
# print(list(loaded.signatures.keys()))  # ["serving_default"]
#
# infer = loaded.signatures["serving_default"]
# print(infer.structured_outputs)
#
# labeling = infer(tf.constant(x))[pretrained_model.output_names[0]]
#
# decoded = imagenet_labels[np.argsort(labeling)[0,::-1][:5]+1]
#
# print("Result after saving and loading:\n", decoded)

pretrained_model.save('savemn', save_format='tf')
loaded = tf.keras.models.load_model('savemn')
print("before loaded model infer ========================")
#loaded(tf.constant(x))
labeling = loaded(tf.constant(x))[0]
decoded = imagenet_labels[np.argsort(labeling)[::-1][:5]+1]
print("Result after saving and loading:\n", decoded)
print("after loaded model infer ========================")
infer = loaded.signatures["serving_default"]
labeling = infer(tf.constant(x))[pretrained_model.output_names[0]]
decoded = imagenet_labels[np.argsort(labeling)[0,::-1][:5]+1]
print("Result after saving and loading:\n", decoded)
