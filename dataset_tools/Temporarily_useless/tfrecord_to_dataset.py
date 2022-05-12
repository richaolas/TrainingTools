import cv2
import fire
import numpy as np
import tensorflow as tf

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

dics = {'image/filename': tf.io.VarLenFeature(tf.string),
        'image/encoded': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'image/width': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        'image/height': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32)}


def parse_tf(example_proto):
    '''

    Args:
        example_proto: protobuf protocol object

    Returns:

    '''
    parse_example = tf.io.parse_single_example(serialized=example_proto, features=dics)
    filename = parse_example['image/filename']
    image = tf.image.decode_jpeg(parse_example['image/encoded'])

    image = tf.concat([image, image, image], axis=2)

    w = parse_example['image/width']
    h = parse_example['image/height']
    text = parse_example['image/object/class/text']
    label = parse_example['image/object/class/label']
    xmin = parse_example['image/object/bbox/xmin']
    xmax = parse_example['image/object/bbox/xmax']
    ymin = parse_example['image/object/bbox/ymin']
    ymax = parse_example['image/object/bbox/ymax']
    tf.print('test', xmin.values, xmin)
    # boxs = tf.stack([xmin.values, xmax.values, ymin.values, ymax.values], axis=1)
    boxs = tf.stack([ymin.values, xmin.values, ymax.values, xmax.values], axis=1)
    tf.print(boxs.shape)
    # boxes = list(np.stack((ymin.values, xmin.values, ymax.values, xmax.values), axis=1))
    #
    # return image, w, h, zip(xmin, xmax, ymin, ymax)
    return image, w, h, boxs, text.values, label.values


def parse_tf_to_image_lable(example_proto):
    parse_example = tf.io.parse_single_example(serialized=example_proto, features=dics)
    # return parse_example
    filename = parse_example['image/filename']
    image = parse_example['image/encoded']  # tf.decode_raw(parse_example['image/encoded'],out_type=tf.uint8)
    image = tf.image.decode_jpeg(image)

    image = tf.concat([image,image,image],axis=2)
    tf.print('##', tf.shape(image))

    w = parse_example['image/width']
    h = parse_example['image/height']
    # print("image size: %d %d" % (w.numpy(), h.numpy()))
    text = parse_example['image/object/class/text']
    label = parse_example['image/object/class/label']
    xmin = parse_example['image/object/bbox/xmin']
    xmax = parse_example['image/object/bbox/xmax']
    ymin = parse_example['image/object/bbox/ymin']
    ymax = parse_example['image/object/bbox/ymax']
    #tf.print('#####', xmin.values, xmin, xmin.shape, xmin.values.shape)
    # boxes = tf.stack([xmin.values, xmax.values, ymin.values, ymax.values], axis=1)
    # boxes：指需要划分的区域，输入格式为 [[ymin,xmin,ymax,xmax]] (要注意！这是一个二维列表)
    # [[x1,x1,x1...],[x2,x2,x2...]] :   1 * k  ----> 1 * 4 * k [[0. 0. 1. 1.],[0. 0. 1. 1.]...]  k * 4
    # N * [x1,y1,x2,y2]
    # xmin: [x1,x2,x3,x4,x5...] [ N ]  ---> [N x 4]

    # 这里注意 [1xN] 和 [N] 的区别
    boxes = tf.stack([ymin.values, xmin.values, ymax.values, xmax.values], axis=1)
    # boxes = list(np.stack((ymin.values, xmin.values, ymax.values, xmax.values), axis=1))
    #
    # return image, w, h, zip(xmin, xmax, ymin, ymax)

    return image, label.values, boxes


def tfrecord_to_dataset(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(parse_tf_to_image_lable)
    return dataset


def main(tfrecord_file=r'C:\simpleSample\demo0_9_nobg.tfrecord'):
    dataset = tf.data.TFRecordDataset(tfrecord_file)  # 读取 TFRecord 文件
    # dataset = dataset.repeat() # 重复数据集
    dataset = dataset.map(parse_tf)  # 解析数据

    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, (image, w, h, boxs, text, label) in enumerate(dataset):
        imgdata = image.numpy()[:, :, ::-1]  # BGR --> RGB
        print("SHAPE:", imgdata.shape)
        w = w.numpy()
        h = h.numpy()
        # print(w, h)
        print(boxs.numpy())
        imgdata = imgdata.astype(np.uint8)
        for idx, (ymin, xmin, ymax, xmax) in enumerate(boxs.numpy()):
            # print(float(x1 * w))
            x1, x2, y1, y2 = xmin, xmax, ymin, ymax
            #print('#', w, h, '(', int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h), ')',
            #      bytes.decode(text[idx].numpy()))

            imgdata = cv2.rectangle(imgdata, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (255, 0, 0), 2)
            imgdata = cv2.putText(imgdata, bytes.decode(text[idx].numpy()), \
                                  (int(x1 * w), int((int(y1 * h) + int(y2 * h)) / 2)), font, 0.8, (0, 0, 0), 2)

        cv2.imshow('', imgdata)
        cv2.waitKey(1)


def main2(tfrecord_file):
    dataset = tfrecord_to_dataset(tfrecord_file)
    for idx, (image, label, boxes) in enumerate(dataset):
        print(image.shape, label.numpy())
        print(boxes)


if __name__ == '__main__':
    fire.Fire(main)
