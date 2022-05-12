# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record_my.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

# absl 库全称是 Abseil Python Common Libraries。它原本是个C++库，后来被迁移到了Python上。
from absl import app, flags, logging

from pathlib import Path
import cv2

# import tensorflow.compat.v1 as tf
# FLAGS = tf.app.flags.FLAGS

# from dataset_tools.utils import utils.dataset_util
# from object_detection.utils import label_map_util

# from utils.dataset_util import *
# from utils.label_map_util import *
import utils.dataset_util as dataset_util
import utils.label_map_util as label_map_util

# flags = tf.app.flags
# flags = tf.compat.v1.flags

flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('name', '', 'Desired challenge name.')
# flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
#                                     'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')

flags.DEFINE_string('label_map_path', 'label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                                                          'difficult instances')
# flags.DEFINE_string('action', 'tfrecord', 'Action in [tfrecord, imageset]')
# flags.DEFINE_string('imageset', 'image', 'image set name')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')

FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']


# YEARS = ['VOC2007', 'VOC2012', 'merged']

def get_all_annotations(annotations_dir):
    all_annotations = []
    for root, dirs, files in os.walk(annotations_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.xml':
                all_annotations.append(os.path.splitext(file)[0])
    return all_annotations


def write_annotations(annotations, data_folder, image_set):
    imageset_main_dir = os.path.join(data_folder, 'ImageSets', 'Main')
    if not os.path.exists(imageset_main_dir):
        os.makedirs(imageset_main_dir)
    imageset_file = imageset_main_dir + os.sep + image_set + '.txt'
    with open(imageset_file, 'w') as wf:
        for annotation in annotations:
            wf.write(annotation + '\n')


def gen_image_set(data_folder, imageset):
    imageset_main_dir = os.path.join(data_folder, 'ImageSets', 'Main')
    if not os.path.exists(imageset_main_dir):
        os.makedirs(imageset_main_dir)
    imageset_file = imageset_main_dir + os.sep + imageset + '.txt'
    annotations_dir = os.path.join(data_folder, FLAGS.annotations_dir)
    with open(imageset_file, 'w') as wf:
        for root, dirs, files in os.walk(annotations_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.xml':
                    wf.write(os.path.splitext(file)[0] + '\n')


def convert_image_format(full_path, dst_fmt='.jpg'):
    # path = 'G:/Samples/Customer/officeflower/JPEGImages/0016.bmp'
    filepath, tmpfilename = os.path.split(full_path)
    shotname, extension = os.path.splitext(tmpfilename)
    print(filepath, shotname, extension)
    if extension != '.jpg':
        dst = os.path.join(filepath, shotname + dst_fmt)
        img = cv2.imread(full_path)
        cv2.imwrite(dst, img)
        return dst
    return full_path


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw .data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
      image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image .data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by .data['filename'] is not a valid JPEG
    """
    img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
    full_path = os.path.join(dataset_directory, img_path)
    if not Path(full_path).exists():
        full_path = data['path']  # for label image tools
    if not Path(full_path).exists():
        full_path = os.path.join(dataset_directory, 'JPEGImages', data['filename'])
    if not Path(full_path).exists():
        logging.error('Image not find at %s', full_path)
        return

    ## convert image format
    full_path = convert_image_format(full_path)

    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    if 'object' in data:
        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue

            difficult_obj.append(int(difficult))

            xmin_val = float(obj['bndbox']['xmin'])
            ymin_val = float(obj['bndbox']['ymin'])
            xmax_val = float(obj['bndbox']['xmax'])
            ymax_val = float(obj['bndbox']['ymax'])
            if xmin_val > xmax_val:
                xmin_val, xmax_val = xmax_val, xmin_val
            if ymin_val > ymax_val:
                ymin_val, ymax_val = ymax_val, ymin_val

            xmin.append(xmin_val / width)
            ymin.append(ymin_val / height)
            xmax.append(xmax_val / width)
            ymax.append(ymax_val / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def main(_):
    data_dir = FLAGS.data_dir
    name = FLAGS.name
    data_folder = os.path.join(data_dir, name)
    annotations_dir = os.path.join(data_dir, name, FLAGS.annotations_dir)
    all_annotations = get_all_annotations(annotations_dir)

    train_set = os.path.join(data_folder, "ImageSets", "Main") + os.sep + 'train.txt'
    val_set = os.path.join(data_folder, "ImageSets", "Main") + os.sep + 'val.txt'

    if not os.path.exists(train_set):
        logging.info("Train set not fount, generate 80% from all data.")
        write_annotations(all_annotations[:int(len(all_annotations) * 0.8)], data_folder, 'train')

    if not os.path.exists(val_set):
        logging.info("Validate set not fount, generate 20% from all data.")
        write_annotations(all_annotations[int(len(all_annotations) * 0.8):], data_folder, 'val')

    label_map_path = FLAGS.label_map_path
    if not os.path.exists(label_map_path):
        logging.info("%s not fount, try to find at %s", label_map_path, data_folder)
        label_map_path = os.path.join(data_folder, FLAGS.label_map_path)
        if not os.path.exists(label_map_path):
            logging.info("%s not fount, failed!", label_map_path)
            return

    output_path = FLAGS.output_path
    if not output_path:
        out_name = os.path.basename(data_folder)
        if not out_name:
            out_name = os.path.basename(data_dir)
        output_path = os.path.basename(out_name) + '.tfrecord'

    logging.info("Using label map path: %s.", label_map_path)
    logging.info("Using annotations dir: %s.", annotations_dir)
    logging.info("Using output path: %s.", output_path)
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)

    # print(FLAGS.data_dir)
    # if FLAGS.set not in SETS:
    #     raise ValueError('set must be in : {}'.format(SETS))
    # if FLAGS.year not in YEARS:
    #    raise ValueError('year must be in : {}'.format(YEARS))

    # years = ['VOC2007', 'VOC2012']
    # if FLAGS.year != 'merged':
    # years = [FLAGS.year]

    # ACTIONSET = ['tfrecord', 'imageset']
    # if FLAGS.action not in ACTIONSET:
    #     raise ValueError('action must be in : {}'.format(ACTIONSET))
    # if FLAGS.action == 'tfrecord':
    #     pass
    # elif FLAGS.action == 'imageset':
    #     gen_image_set(FLAGS.data_dir, FLAGS.year, FLAGS.imageset)
    #     return

    for set_name, image_set_path in zip(('train', 'val'), (train_set, val_set)):
        logging.info("Generate data set %s in %s.", set_name, image_set_path)

        examples_list = dataset_util.read_examples_list(image_set_path)
        writer = tf.io.TFRecordWriter(
            os.path.splitext(output_path)[0] + '_' + set_name + os.path.splitext(output_path)[1])
        step = max(len(examples_list) // 10 // 100 * 100, 10)
        for idx, example in enumerate(examples_list):
            if idx % step == 0:
                logging.info('On image %d of %d', idx, len(examples_list))
            path = os.path.join(annotations_dir, example + '.xml')
            if not Path(path).exists():
                logging.error('Annotation xml %s not exist, press any key to continue..., q for quit.', path)
                key = input()
                if key == 'q':
                    break
                else:
                    continue

            with tf.io.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str.encode('utf-8'))
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            # logging.info("Create tf example for %s.", path)

            tf_example = dict_to_tf_example(data, data_folder, label_map_dict,
                                            FLAGS.ignore_difficult_instances)
            if tf_example:
                writer.write(tf_example.SerializeToString())

        writer.close()


"""
usage
data_dir: root data folder
name:     task sample name (在data_dir下面的文件夹，默认为空，代表 data_dir 就是数据集目录，而不是所有数据根目录)

如果 data_dir/name/ImageSets/Main 下面没有 train.txt 或者 val.txt 则根据 4:1 比例自动生成
label_map_dict： 相对于 data_dir/name/，路径为 data_dir/name/label_map_dict

分成多个时间(year)文件夹的好处是，可以增量放数据，

"""

"""
release note:
2022/02/17:
数据集格式：根目录, label_map.pbtxt 放在这个目录下面
下面命令会自动生成 xxx_train.tfrecord xxx_val.tfrecord
python create_pascal_tf_record.py --data_dir=data_folder  --output_path=xxx.tfrecord

"""

# ----------------
# python create_pascal_tf_record_my.py --action=imageset --data_dir=C:\simpleSample --year=doorline --set=train
# python create_pascal_tf_record_my.py --action=tfrecord --data_dir=C:\simpleSample --label_map_path=.\.data\test.pbtxt --year=doorline --imageset=image --set=train --output_path=C:\pascal_train.record
# python create_pascal_tf_record_my.py --data_dir=G:\dataset\VOCdevkit --label_map_path=pascal_label_map.pbtxt --year=VOC2012 --set=train --output_path=G:\pascal_train.record
# >python create_pascal_tf_record_my.py --data_dir=H:\Samples\VOC\VOCdevkit --label_map_path=.
# \data\pascal_label_map.pbtxt --year=VOC2012 --imageset=train --set=train --output_path=H:\pascal_train.record
if __name__ == '__main__':
    app.run(main)
