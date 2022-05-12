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
The process will generate image_set.txt auto, if you want to re-generate the set file.
You can delete the image_set.txt manual.
If you need create different dateset by image set text name, you need to split yourself.
Or you can split the tfrecord after the dataset has been loaded.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record_my.py \
        --data_dir=/home/user/VOCdevkit \
        --year=2020,2021 \
        --output_path=/home/user/pascal.record


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import random

from lxml import etree
import PIL.Image
import tensorflow as tf

from absl import app, flags, logging

from pathlib import Path

# import tensorflow.compat.v1 as tf
# FLAGS = tf.app.flags.FLAGS

# from dataset_tools.utils import utils.dataset_util
# from object_detection.utils import label_map_util

# from utils.dataset_util import *
# from utils.label_map_util import *
import utils.dataset_util as dataset_util
import utils.label_map_util as label_map_util

flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('year', 'VOC2007', 'Desired challenge year.')
# flags.DEFINE_string('project', '', 'Desired project name.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', '',
                    'Path to label map proto')  # './data/pascal_label_map.pbtxt'
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                                                          'difficult instances')

flags.DEFINE_string('action', 'tfrecord', 'Action in [tfrecord, imageset]')
# flags.DEFINE_string('imageset', 'image', 'image set name')

FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']


# YEARS = ['VOC2007', 'VOC2012', 'merged']


def gen_image_set(data_dir, year):
    """generate image set text file from annotation xmls
    Args:
        data_dir:
        year:
        image_set:

    Returns:

    """
    image_set_main_dir = os.path.join(data_dir, year, 'ImageSets', 'Main')
    if not os.path.exists(image_set_main_dir):
        os.makedirs(image_set_main_dir)
    image_set_file = image_set_main_dir + os.sep + FLAGS.set + '.txt'
    if os.path.exists(image_set_file):
        return image_set_file

    annotations_dir = os.path.join(data_dir, year, FLAGS.annotations_dir)
    with open(image_set_file, 'w') as wf:
        for root, dirs, files in os.walk(annotations_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.xml':
                    wf.write(os.path.splitext(file)[0] + '\n')
    return image_set_file


def dict_to_tf_example(data,
                       dataset_directory,
                       year,
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
        full_path = os.path.join(dataset_directory, year, 'JPEGImages', data['filename'])
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

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))
            break  #[todo]

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
    logging.info('Prepare process samples in {}'.format(FLAGS.data_dir))
    data_dir = FLAGS.data_dir

    years = list(map(lambda x: x.strip(), str(FLAGS.year).split(',')))
    label_map_file = FLAGS.label_map_path
    if not os.path.exists(label_map_file):
        label_map_file = os.path.join(data_dir, 'label_map.pbtxt')
        if not os.path.exists(label_map_file):
            raise FileExistsError('label map file not exist.')

    label_map_dict = label_map_util.get_label_map_dict(label_map_file)

    # output path
    output_path = FLAGS.output_path
    if not output_path:
        output_path = '.'  # os.path.basename(os.path.dirname(data_dir+os.sep)) + '.tfrecord'
    logging.info('Prepare write samples to {}'.format(output_path))

    # 先默认比例 6:2:2 train valid test
    sample_name = os.path.basename(os.path.dirname(data_dir + os.sep))
    output_train = output_path + os.sep + sample_name + '_train.tfrecord'
    output_valid = output_path + os.sep + sample_name + '_valid.tfrecord'
    output_test = output_path + os.sep + sample_name + '_test.tfrecord'

    writers = {
        output_train: tf.io.TFRecordWriter(output_train),
        output_valid: tf.io.TFRecordWriter(output_valid),
        output_test: tf.io.TFRecordWriter(output_test),
    }

    for year in years:
        logging.info('Reading from PASCAL %s dataset.', year)

        examples_path = gen_image_set(FLAGS.data_dir, year)
        examples_list = dataset_util.read_examples_list(examples_path)

        annotations_dir = os.path.join(data_dir, year, FLAGS.annotations_dir)

        for idx, example in enumerate(examples_list):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(examples_list))
            path = os.path.join(annotations_dir, example + '.xml')
            with tf.io.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str.encode('utf-8'))
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            tf_example = dict_to_tf_example(data, FLAGS.data_dir, year, label_map_dict,
                                            FLAGS.ignore_difficult_instances)

            random_val = random.randint(1, 100)
            writer = writers[output_train]

            if 60 < random_val <= 80:
                writer = writers[output_valid]
            elif 80 <= random_val:
                writer = writers[output_test]

            writer.write(tf_example.SerializeToString())

    for writer in writers.values():
        writer.close()


if __name__ == '__main__':
    app.run(main)

"""
python pascal_create_project_tf_record_train_valid_test.py --data_dir=G:\Samples\VOC_CUSTOM\demoproject\ --year=2020,2021 --set=train

"""
