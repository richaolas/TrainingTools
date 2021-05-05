r"""Show the tfrecord file

Example usage:
    python pascal_tf_record_view.py data.tfrecord object_name

if do not supply a object_name, then all objects will be show


"""
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

def parse_tf_example(example_proto):
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
    # boxs = tf.stack([xmin.values, xmax.values, ymin.values, ymax.values], axis=1)
    boxs = tf.stack([ymin.values, xmax.values, xmin.values, ymax.values], axis=1)
    # boxes = list(np.stack((ymin.values, xmin.values, ymax.values, xmax.values), axis=1))
    #
    # return image, w, h, zip(xmin, xmax, ymin, ymax)
    return filename, image, w, h, boxs, text.values, label.values


def main(tfrecord_file, object_show=None):
    dataset = tf.data.TFRecordDataset(tfrecord_file)  # 读取 TFRecord 文件
    # dataset = dataset.repeat() # 重复数据集
    dataset = dataset.map(parse_tf_example)  # 解析数据

    font = cv2.FONT_HERSHEY_SIMPLEX

    wait_time = 0

    for idx, (filename, image, w, h, boxs, text, label) in enumerate(dataset):
        print(idx, (filename, image, w, h, boxs, text, label))
        if object_show:
            all_objects = []
            for i in range(len(text)):
                text_string = bytes.decode(text[i].numpy())
                all_objects.append(text_string)

            if object_show not in all_objects:
                continue

        imgdata = image.numpy()[:, :, ::-1].copy()
        w = w.numpy()
        h = h.numpy()
        for idx, (x1, x2, y1, y2) in enumerate(boxs.numpy()):
            x1, x2, y1, y2 = y1, x2, x1, y2
            # cv2.rectangle(imgdata, (x1 * w, y1 * h), (x2 * w, y2 * h), (0,255,0), 2)
            text_string = bytes.decode(text[idx].numpy())
            print(text_string)

            if object_show and object_show != text_string:
                continue

            imgdata = cv2.rectangle(imgdata, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (255, 0, 0), 2)
            imgdata = cv2.putText(imgdata, text_string, \
                                  (int((int(x1 * w) + int(x2 * w)) / 2), int((int(y1 * h) + int(y2 * h)) / 2)), font,
                                  0.6, (0, 0, 0), 2)

        filename_str = bytes.decode(tf.sparse.to_dense(filename).numpy()[0])
        cv2.putText(imgdata, filename_str, (0, 10), font, 0.4, (0, 0, 0), 1)
        cv2.imshow('VOC Viewer', imgdata)
        key = cv2.waitKey(wait_time)
        if key == ord('q'):
            break
        elif key == ord('p'):
            wait_time = (wait_time + 1) % 2

    cv2.destroyAllWindows()


if __name__ == '__main__':
    fire.Fire(main)
