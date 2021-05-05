import hashlib
import io
import logging
import os
import numpy as np
import cv2

from lxml import etree
import PIL.Image
import tensorflow as tf

from absl import app, flags, logging
import fire
from pathlib import Path

import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse

import utils.dataset_util as dataset_util
import utils.label_map_util as label_map_util

#def dict_to_tf_example(data_dict, dataset_directory):

# save the whole image, and the roi crop image is the object which need to classify

def bmpToJpg(file_path):
   for fileName in os.listdir(file_path):
       print(fileName)
       newFileName = fileName[0:fileName.find(".bmp")]+".jpg"
       print(newFileName)


       im = PIL.Image.open(file_path+"\\"+fileName)
       im.save(file_path+"\\"+newFileName)


def bmpToJpg(file_path, fileName):
    print(fileName)
    newFileName = fileName[0:fileName.find(".bmp")] + ".jpg"
    print(newFileName)

    image_np = cv2.imread(file_path + os.sep + fileName) # 用opencv 打开再存储，为了防止单通道图像的问题
    #im = PIL.Image.open(file_path + "\\" + fileName)
    newpath = file_path + "\\" + newFileName
    #im.save(newpath)
    cv2.imwrite(newpath, image_np)
    return newpath

def deleteImages(file_path, imageFormat):
   command = "del "+file_path+"\\*."+imageFormat
   os.system(command)

def dict_to_tf_example(data_dict, dataset_directory):
    #img_path = os.path.join(data_dict['folder'], image_subdirectory, data_dict['filename'])
    #global roi
    full_path = data_dict['filename']
    if not Path(full_path).exists():
        full_path = os.path.join(dataset_directory, full_path)  # for label image tools

    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
        # if roi:
        #     encoded_jpg_io = io.BytesIO(encoded_jpg)
        #     image = PIL.Image.open(encoded_jpg_io)
        #     image = image.crop(roi)
        #     image.save('.temp.jpg')
        #     with tf.io.gfile.GFile('.temp.jpg', 'rb') as tmp_fid:
        #         encoded_jpg = tmp_fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)

    if image.format != 'JPEG':
        if image.format == "BMP":
            newJPEGPath = bmpToJpg(dataset_directory, data_dict['filename'])
            with tf.io.gfile.GFile(newJPEGPath, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = PIL.Image.open(encoded_jpg_io)
            os.remove(newJPEGPath) # delete generate tmp file
        else:
            raise ValueError('Image format not JPEG or BMP')

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

    roi = None
    #roi = (300, 100, 300 + 320, 100 + 300)

    if not roi:
        roi = (0, 0, image.width, image.height)

    data_dict = {
        'width':image.width,
        'height':image.height,
        'filename':os.path.join(text, image_name),
        'objects': [{
            'xmin':roi[0],
            'xmax':roi[2],
            'ymin':roi[1],
            'ymax':roi[3],
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
                    #print(data_dict)
                    example = dict_to_tf_example(data_dict, data_dir)
                    writer.write(example.SerializeToString())
                label_idx += 1


if __name__ == '__main__':
    fire.Fire(main)
