import numpy as np
from PIL import Image
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from tensorflow.keras import layers, datasets


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


objs = unpickle(r'G:\dataset\cifar-10-batches-py\data_batch_1')
print(objs)

img = objs[b'.data'][0]
img = img.reshape((3, 32, 32)).transpose(1, 2, 0)[:, :, ::-1]
# # [rrrr][gggg][bbbb]
# # [rgb][rgb]
# r = img[0]
# g = img[1]
# b = img[2]
# img = cv2.merge((b,g,r)).reshape((32,32,-1))
img_cv = cv2.resize(img, (224, 244))
cv2.imshow('', img_cv)
cv2.waitKey(0)
'''
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
'''


# img = img.reshape(32,32,-1)


def resize(d, size=(224, 224)):
    return np.array([np.array(Image.fromarray(v).resize(size, Image.ANTIALIAS))
                     for i, v in enumerate(d)])


(train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()
print(train_x.shape)
for i in range(1):
    image = Image.fromarray(train_x[i])  # Image.frombytes
    img_cv = cv2.resize(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), (224, 244))
    cv2.imshow('', img_cv)
    cv2.waitKey(0)

# train_x, test_x = resize(train_x[:10000])/255.0, resize(test_x)/255.0
# train_y = train_y[:10000]
