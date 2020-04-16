import tensorflow as tf
import cv2
import numpy as np
import fire

#
# #其中的参数read_and_decode就是我们定义的解析example的函数，实现如下：
# feature_description = {
#         'image/height': tf.io.FixedLenFeature([], tf.int64),
#         'image/width': tf.io.FixedLenFeature([], tf.int64),
#         'image/filename': tf.io.FixedLenFeature([], tf.string),
#         'image/source_id': tf.io.FixedLenFeature([], tf.string),
#         'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
#         'image/encoded': tf.io.FixedLenFeature([], tf.string),
#         'image/format': tf.io.FixedLenFeature([], tf.string),
#         'image/object/bbox/xmin': tf.io.FixedLenSequenceFeature([], tf.float32),
#         'image/object/bbox/xmax': tf.io.FixedLenSequenceFeature([], tf.float32),
#         'image/object/bbox/ymin': tf.io.FixedLenSequenceFeature([], tf.float32),
#         'image/object/bbox/ymax': tf.io.FixedLenSequenceFeature([], tf.float32),
#         'image/object/class/text': tf.io.FixedLenSequenceFeature([], tf.string),
#         'image/object/class/label': tf.io.FixedLenSequenceFeature([], tf.float32),
#         'image/object/difficult': tf.io.FixedLenFeature([], tf.int64),
#         'image/object/truncated': tf.io.FixedLenFeature([], tf.int64),
#         'image/object/view': tf.io.FixedLenFeature([], tf.string),
#     }

# feature_description = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
#     'image': tf.io.FixedLenFeature([], tf.string),
#     'label': tf.io.FixedLenFeature([], tf.string)
# }

def parse_tf(example_proto):
    dics = {}

    dics['image/filename'] = tf.io.VarLenFeature(tf.string)
    dics['image/encoded'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    dics['image/width'] = tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
    dics['image/height'] = tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
    dics['image/object/class/text'] = tf.io.VarLenFeature(tf.string)
    dics['image/object/class/label'] = tf.io.VarLenFeature(tf.int64)
    dics['image/object/bbox/xmin'] = tf.io.VarLenFeature(tf.float32)
    dics['image/object/bbox/xmax'] = tf.io.VarLenFeature(tf.float32)
    dics['image/object/bbox/ymin'] = tf.io.VarLenFeature(tf.float32)
    dics['image/object/bbox/ymax'] = tf.io.VarLenFeature(tf.float32)

    parse_example = tf.io.parse_single_example(serialized=example_proto, features=dics)
    # return parse_example
    filename = parse_example['image/filename']
    image = parse_example['image/encoded']  # tf.decode_raw(parse_example['image/encoded'],out_type=tf.uint8)
    image = tf.image.decode_jpeg(image)
    w = parse_example['image/width']
    h = parse_example['image/height']

    text = parse_example['image/object/class/text']
    label = parse_example['image/object/class/label']
    xmin = parse_example['image/object/bbox/xmin']
    xmax = parse_example['image/object/bbox/xmax']
    ymin = parse_example['image/object/bbox/ymin']
    ymax = parse_example['image/object/bbox/ymax']
    print(xmin.values, xmin)
    boxs = tf.stack([xmin.values, xmax.values, ymin.values, ymax.values], axis=1)
    # boxes = list(np.stack((ymin.values, xmin.values, ymax.values, xmax.values), axis=1))
    #
    # return image, w, h, zip(xmin, xmax, ymin, ymax)
    return image, w, h, boxs, text.values, label.values


def main(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file)  # 读取 TFRecord 文件
    # dataset = dataset.repeat() # 重复数据集
    dataset = dataset.map(parse_tf)  # 解析数据
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, (image, w, h, boxs, text, label) in enumerate(dataset):
        imgdata = image.numpy()[:, :, ::-1]
        w = w.numpy()
        h = h.numpy()
        for idx, (x1, x2, y1, y2) in enumerate(boxs.numpy()):
            print(float(x1 * w))
            # cv2.rectangle(imgdata, (x1 * w, y1 * h), (x2 * w, y2 * h), (0,255,0), 2)

            imgdata = cv2.rectangle(imgdata, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (255, 0, 0), 2)
            imgdata = cv2.putText(imgdata, bytes.decode(text[idx].numpy()), \
                                  (int(x1 * w), int((int(y1 * h) + int(y2 * h)) / 2)), font, 0.8, (0, 0, 0), 2)
        cv2.imshow('', imgdata)
        cv2.waitKey(0)


if __name__ == '__main__':
    fire.Fire(main)

