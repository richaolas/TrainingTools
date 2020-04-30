import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

from absl import app, flags, logging
import fire
from pathlib import Path

import utils.dataset_util as dataset_util
import utils.label_map_util as label_map_util

#def dict_to_tf_example(data_dict, dataset_directory):

roi = None
roi = (320, 80, 320+300, 80 + 200)

def dict_to_tf_example(data_dict, dataset_directory):
    #img_path = os.path.join(data_dict['folder'], image_subdirectory, data_dict['filename'])
    global roi
    full_path = data_dict['filename']
    if not Path(full_path).exists():
        full_path = os.path.join(dataset_directory, full_path)  # for label image tools
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
        if roi:
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = PIL.Image.open(encoded_jpg_io)
            image = image.crop(roi)
            image.save('.temp.jpg')
            with tf.io.gfile.GFile('.temp.jpg', 'rb') as tmp_fid:
                encoded_jpg = tmp_fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    width = image.width #data_dict.get('width', image.width)
    height = image.height #data_dict.get('height', image.height)
    filename = data_dict['filename']
    source_id = data_dict.get('source_id', filename)
    sha256 = hashlib.sha256(encoded_jpg).hexdigest()
    format = 'jpeg'
    
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    # 至少需要一个
    for obj in data_dict.get('objects', []):
        xmin.append(float(obj['xmin']) / width)
        ymin.append(float(obj['ymin']) / height)
        xmax.append(float(obj['xmax']) / width)
        ymax.append(float(obj['ymax']) / height)
        classes_text.append(obj['text'].encode('utf8'))
        classes.append(obj['label'])
        difficult_obj.append(obj.get('difficult', 0))
        truncated.append(obj.get('truncated', 0))
        poses.append(obj.get('pose', 'Unspecified').encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(source_id.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(sha256.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(format.encode('utf8')),
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

def make_classification_dict(data_dir, image_name, text, label):
    image_path = os.path.join(data_dir, text, image_name)
    image = PIL.Image.open(image_path)
    data_dict = {
        'width':image.width,
        'height':image.height,
        'filename':os.path.join(text, image_name),
        'objects': [{
            'xmin':0,
            'xmax':image.width,
            'ymin':0,
            'ymax':image.height,
            'text':text,
            'label':label
        }]
    }
    return data_dict


def main(data_dir, output_path):
    print(data_dir)
    with tf.io.TFRecordWriter(output_path) as writer:
        for home, dirs, files in os.walk(data_dir):
            label_idx = 0
            for dir in dirs: # loop all class
                label_text = dir
                fullname = os.path.join(home, dir)
                for file in os.listdir(fullname):
                    data_dict = make_classification_dict(data_dir, file, label_text, label_idx)
                    print(data_dict)
                    example = dict_to_tf_example(data_dict, data_dir)
                    writer.write(example.SerializeToString())
                label_idx += 1


if __name__ == '__main__':
    fire.Fire(main)

#
#
# def gen_image_set(data_dir, year, imageset):
#     imageset_main_dir = os.path.join(data_dir, year, 'ImageSets', 'Main')
#     if not os.path.exists(imageset_main_dir):
#         os.makedirs(imageset_main_dir)
#     imageset_file = imageset_main_dir + os.sep + imageset + '_' + FLAGS.set + '.txt'
#     annotations_dir = os.path.join(data_dir, year, FLAGS.annotations_dir)
#     with open(imageset_file, 'w') as wf:
#         for root, dirs, files in os.walk(annotations_dir):
#             for file in files:
#                 if os.path.splitext(file)[1] == '.xml':
#                     wf.write(os.path.splitext(file)[0] + '\n')
#

#
#
# def main(_):
#     print(FLAGS.data_dir)
#     if FLAGS.set not in SETS:
#         raise ValueError('set must be in : {}'.format(SETS))
#     #if FLAGS.year not in YEARS:
#     #    raise ValueError('year must be in : {}'.format(YEARS))
#
#     data_dir = FLAGS.data_dir
#     #years = ['VOC2007', 'VOC2012']
#     #if FLAGS.year != 'merged':
#     years = [FLAGS.year]
#
#     ACTIONSET = ['tfrecord', 'imageset']
#     if FLAGS.action not in ACTIONSET:
#         raise ValueError('action must be in : {}'.format(ACTIONSET))
#     if FLAGS.action == 'tfrecord':
#         pass
#     elif FLAGS.action == 'imageset':
#         gen_image_set(FLAGS.data_dir, FLAGS.year, FLAGS.imageset)
#         return
#
#     writer = tf.io.TFRecordWriter(FLAGS.output_path)
#
#     label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
#
#     for year in years:
#         logging.info('Reading from PASCAL %s dataset.', year)
#         examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main',
#                                      FLAGS.imageset + '_' + FLAGS.set + '.txt')
#         annotations_dir = os.path.join(data_dir, year, FLAGS.annotations_dir)
#         examples_list = dataset_util.read_examples_list(examples_path)
#         for idx, example in enumerate(examples_list):
#             if idx % 100 == 0:
#                 logging.info('On image %d of %d', idx, len(examples_list))
#             path = os.path.join(annotations_dir, example + '.xml')
#             with tf.io.gfile.GFile(path, 'r') as fid:
#                 xml_str = fid.read()
#             xml = etree.fromstring(xml_str)
#             data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
#
#             tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
#                                             FLAGS.ignore_difficult_instances)
#             writer.write(tf_example.SerializeToString())
#
#     writer.close()
#
# #----------------
# # python create_pascal_tf_record.py --action=imageset --data_dir=C:\simpleSample --year=doorline --set=train
# # python create_pascal_tf_record.py --action=tfrecord --data_dir=C:\simpleSample --label_map_path=.\data\test.pbtxt --year=doorline --imageset=image --set=train --output_path=C:\pascal_train.record
# # python create_pascal_tf_record.py --data_dir=G:\dataset\VOCdevkit --label_map_path=pascal_label_map.pbtxt --year=VOC2012 --set=train --output_path=G:\pascal_train.record
# #
# if __name__ == '__main__':
#     app.run(main)
