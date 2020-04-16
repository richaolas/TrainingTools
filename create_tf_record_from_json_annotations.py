import fire
import json

from io import StringIO
from io import BytesIO
import tensorflow as tf
import os

import numpy as np
import IPython.display as display

from PIL import Image

# The following functions can be used to convert a value to a type compatible
# with tf.Example.

# def _bytes_feature(value):
#   """Returns a bytes_list from a string / byte."""
#   if isinstance(value, type(tf.constant(0))):
#     value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
#   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
# def _float_feature(value):
#   """Returns a float_list from a float / double."""
#   return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
#
# def _int64_feature(value):
#   """Returns an int64_list from a bool / enum / int / uint."""
#   return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def dict_to_tf_example(samplepath, annotation):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
      image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    # how to save image data, and what format
    image_string = open(samplepath + os.sep + annotation['image'], 'rb').read()
    xmin, xmax, ymin, ymax = [], [], [], []
    width, height = annotation['width'], annotation['height']
    for box in annotation['boxs']:
        xmin.append(box[0]/width)
        xmax.append((box[0]+box[2])/width)
        ymin.append(box[1] / height)
        ymax.append((box[1] + box[3]) / height)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(annotation['height']),
        'image/width': int64_feature(annotation['width']),
        'image/filename': bytes_feature(annotation['image'].encode('utf8')),
        #'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        #'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(image_string),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        'image/object/class/text': bytes_list_feature([s.encode('utf8') for s in annotation['class-text']]),
        'image/object/class/label': int64_list_feature(annotation['class-id']),
        #'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        #'image/object/truncated': dataset_util.int64_list_feature(truncated),
        #'image/object/view': dataset_util.bytes_list_feature(poses),
    }))

    return example

def generate_tfrecord(samplepath, jsonpath, record_file):
    record_file = record_file + '.tfrecords'
    json_file = open(jsonpath, 'r')
    annotations = json.loads(json_file.read())
    with tf.io.TFRecordWriter(record_file) as writer:
        for idx, annotation in enumerate(annotations):
            tf_example = dict_to_tf_example(samplepath, annotation)
            #image_string = open(samplepath + os.sep + annotation['image'], 'rb').read()
            #Image.open(BytesIO(image_string)).show()
            #tf_example = image_example(image_string, label)
            writer.write(tf_example.SerializeToString())
            if idx == 5:
                break

if __name__ == '__main__':
    fire.Fire(generate_tfrecord)