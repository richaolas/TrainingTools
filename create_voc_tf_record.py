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
    python object_detection/dataset_tools/create_pascal_tf_record4raccoon.py \
	--data_dir=/home/forest/dataset/raccoon_dataset-master/images \
	--set=/home/forest/dataset/raccoon_dataset-master/train.txt \
	--output_path=/home/forest/dataset/raccoon_dataset-master/train.record \
	--label_map_path=/home/forest/dataset/raccoon_dataset-master/raccoon_label_map.pbtxt \
	--annotations_dir=/home/forest/dataset/raccoon_dataset-master/annotations
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import hashlib
import io
import logging
import os
 
from lxml import etree
import PIL.Image
import tensorflow as tf
 
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
 
# python create_voc_tf_record.py --data_dir=H:/ --data_name=VOC2018 --set=20181219_110106.txt
flags = tf.app.flags
flags.DEFINE_string('data_dir', 'C:/', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('data_name', 'VOC2019bands', 'Desired .data set name')
flags.DEFINE_string('set', '20190315_140146', 'Convert training set, validation set or merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
#flags.DEFINE_string('data_set', '', 'Desired .data set list file name')

flags.DEFINE_string('output_path', r'F:/train.tfrecords', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', r'C:/VOC2019bands/label_map.pbtxt', 'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS
 
SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'VOC2018', 'merged']

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
  with tf.gfile.GFile(full_path, 'rb') as fid:
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
    
      # added by renjch
      if (obj['name'] not in label_map_dict.keys()):
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
  
# check the dataset is supported
#  if FLAGS.set not in SETS:
#    raise ValueError('set must be in : {}'.format(SETS))

  data_names = FLAGS.data_name.split(';')
  if len(data_names) == 0:
      raise ValueError('.data name must be setted')
  print('.data names: ' + str(data_names))
  
  data_set = FLAGS.set
  if FLAGS.set == '':
    raise ValueError('set must be setted') 
  
  data_dir = FLAGS.data_dir

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  for data_name in data_names:
    logging.info('Reading from PASCAL %s dataset.', data_name)

    examples_path = os.path.join(data_dir, data_name, 'ImageSets', 'Main', data_set + '.txt')
    print('Examples path: ', examples_path)
    annotations_dir = os.path.join(data_dir, data_name, FLAGS.annotations_dir)
    
    examples_list = dataset_util.read_examples_list(examples_path)
    print(examples_list)
    for idx, example in enumerate(examples_list):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples_list))
      path = os.path.join(annotations_dir, example + '.xml')
      print(path)
      with tf.gfile.GFile(path, 'r') as fid:
        xml_str = fid.read()
      xml = etree.fromstring(xml_str)
      data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

      tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                      FLAGS.ignore_difficult_instances)
      writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
  


































# 
#def dict_to_tf_example(.data,
#                       dataset_directory,
#                       label_map_dict,
#                       ignore_difficult_instances=False,
#                       image_subdirectory='JPEGImages'):
#  """Convert XML derived dict to tf.Example proto.
#  Notice that this function normalizes the bounding box coordinates provided
#  by the raw .data.
#  Args:
#    .data: dict holding PASCAL XML fields for a single image (obtained by
#      running dataset_util.recursive_parse_xml_to_dict)
#    dataset_directory: Path to root directory holding PASCAL dataset
#    label_map_dict: A map from string label names to integers ids.
#    ignore_difficult_instances: Whether to skip difficult instances in the
#      dataset  (default: False).
#    image_subdirectory: String specifying subdirectory within the
#      PASCAL dataset directory holding the actual image .data.
#  Returns:
#    example: The converted tf.Example.
#  Raises:
#    ValueError: if the image pointed to by .data['filename'] is not a valid JPEG
#  """
#  # 下面这句里的replace就是针对reccoon的标注文件里的filename标签后缀错误而特别添加的
#  img_path = os.path.join(dataset_directory, .data['filename'].replace('.png','.jpg').replace('.PNG','.jpg'))
#  full_path = img_path
#  with tf.gfile.GFile(full_path, 'rb') as fid:
#    encoded_jpg = fid.read()
#  encoded_jpg_io = io.BytesIO(encoded_jpg)
#  image = PIL.Image.open(encoded_jpg_io)
#  if image.format != 'JPEG':
#    raise ValueError('Image format not JPEG')
#  key = hashlib.sha256(encoded_jpg).hexdigest()
# 
#  width = int(.data['size']['width'])
#  height = int(.data['size']['height'])
# 
#  xmin = []
#  ymin = []
#  xmax = []
#  ymax = []
#  classes = []
#  classes_text = []
#  truncated = []
#  poses = []
#  difficult_obj = []
#  
#  for obj in .data['object']:
#    difficult = bool(int(obj['difficult']))
#    if ignore_difficult_instances and difficult:
#      continue
# 
#    difficult_obj.append(int(difficult))
# 
#    xmin.append(float(obj['bndbox']['xmin']) / width)
#    ymin.append(float(obj['bndbox']['ymin']) / height)
#    xmax.append(float(obj['bndbox']['xmax']) / width)
#    ymax.append(float(obj['bndbox']['ymax']) / height)
#    classes_text.append(obj['name'].encode('utf8'))
#    classes.append(label_map_dict[obj['name']])
#    truncated.append(int(obj['truncated']))
#    poses.append(obj['pose'].encode('utf8'))
# 
#  example = tf.train.Example(features=tf.train.Features(feature={
#      'image/height': dataset_util.int64_feature(height),
#      'image/width': dataset_util.int64_feature(width),
#      'image/filename': dataset_util.bytes_feature(
#          .data['filename'].encode('utf8')),
#      'image/source_id': dataset_util.bytes_feature(
#          .data['filename'].encode('utf8')),
#      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
#      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
#      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
#      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
#      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
#      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
#      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
#      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
#      'image/object/class/label': dataset_util.int64_list_feature(classes),
#      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
#      'image/object/truncated': dataset_util.int64_list_feature(truncated),
#      'image/object/view': dataset_util.bytes_list_feature(poses),
#  }))
#  return example
# 
# 
#def main(_):
#  #if FLAGS.set not in SETS:
#  #  raise ValueError('set must be in : {}'.format(SETS))
#  #if FLAGS.year not in YEARS:
#  #  raise ValueError('year must be in : {}'.format(YEARS))
# 
##  data_dir = FLAGS.data_dir
#  years = ['VOC2007', 'VOC2012']
#  if FLAGS.year != 'merged':
#    years = [FLAGS.year]
# 
#  print('here-----------------------' + FLAGS.output_path)
#  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
#  
#  print('here-----------------------')
# 
#  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
#  
#  
# 
#  for year in years:
#    logging.info('Reading from PASCAL %s dataset.', year)
#    examples_path = FLAGS.set
#    #                             'aeroplane_' + FLAGS.set + '.txt')
#   
#    annotations_dir = FLAGS.annotations_dir
#    print('here-----------------------???????????' + examples_path + annotations_dir)
#    examples_list = dataset_util.read_examples_list(examples_path)
#    
#    print('here----------------------xxxxxxxxxx')
#    for idx, example in enumerate(examples_list):
#      if idx % 100 == 0:
#        logging.info('On image %d of %d', idx, len(examples_list))
#      path = os.path.join(annotations_dir, example + '.xml')
#      with tf.gfile.GFile(path, 'r') as fid:
#        xml_str = fid.read()
#      xml = etree.fromstring(xml_str)
#      .data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
# 
#      tf_example = dict_to_tf_example(.data, FLAGS.data_dir, label_map_dict,
#                                      FLAGS.ignore_difficult_instances)
#      writer.write(tf_example.SerializeToString())
# 
#  writer.close()
# 
# 
#if __name__ == '__main__':
#  tf.app.run()
