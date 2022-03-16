""" Sample TensorFlow XML-to-TFRecord converter

usage: generate_tfrecord.py [-h] [-x XML_DIR] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -x XML_DIR, --xml_dir XML_DIR
                        Path to the folder where the input .xml files are stored.
  -l LABELS_PATH, --labels_path LABELS_PATH
                        Path to the labels (.pbtxt) file.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path of output TFRecord (.record) file.
  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        Path to the folder where the input image files are stored. Defaults to the same directory as XML_DIR.
  -c CSV_PATH, --csv_path CSV_PATH
                        Path of output .csv file. If none provided, then no file will be written.
  -t TASK_TYPE, --task_type TASK_TYPE
                        Type of task, in [objectdetect, classify, segment]. Defaults to objectdetect
"""

import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse
from shutil import copyfile
import xml.dom.minidom as DOM
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow XML-to-TFRecord converter")
parser.add_argument("-x",
                    "--xml_dir",
                    help="Path to the folder where the input .xml files are stored.",
                    type=str)
parser.add_argument("-l",
                    "--labels_path",
                    help="Path to the labels (.pbtxt) file.", type=str)
parser.add_argument("-o",
                    "--output_path",
                    help="Path of output TFRecord (.record) file.", type=str)
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored. "
                         "Defaults to the same directory as XML_DIR.",
                    type=str, default=None)
parser.add_argument("-c",
                    "--csv_path",
                    help="Path of output .csv file. If none provided, then no file will be "
                         "written.",
                    type=str, default=None)
parser.add_argument("-t",
                    "--task_type",
                    help="Type of task, in [objectdetect, classify, segment]. Defaults to objectdetect.",
                    type=str, default="ojectdetect")
parser.add_argument("-r",
                    "--roi",
                    help="Type of task, in [objectdetect, classify, segment]. Defaults to objectdetect.",
                    type=str, default="0.0,0.0,1.0,1.0")

args = parser.parse_args()

if args.image_dir is None:
    args.image_dir = args.xml_dir

if args.output_path is None:
    args.output_path = args.image_dir

def generate_classify_labels_file(image_dir):
    default_label_map_name = 'label_map.pbtxt'
    labels_path = os.path.join(image_dir, default_label_map_name)
    with open(labels_path, 'w') as labels_file:
        id = 1 #Label map id 0 is reserved for the background label
        for home, dirs, files in os.walk(image_dir):
            for dir in dirs:
                labels_file.write('item {\n')
                labels_file.write('\tid:{0}\n'.format(id))
                labels_file.write('\tname:\'{0}\'\n'.format(dir))
                labels_file.write('}\n')
                id += 1
    return labels_path

def generate_xml(image_name, image_label, output_path):
    # 生成根节点
    root = ET.Element('annotation')
    # 生成第一个子节点 head
    folder = ET.SubElement(root, 'folder')
    folder.text = ""
    # head 节点的子节点
    filename = ET.SubElement(root, 'filename')
    filename.text = image_name
    # 生成 root 的第二个子节点 body
    path = ET.SubElement(root, 'path')
    path.text = os.path.join(output_path, image_name)
    # body 的内容
    source = ET.SubElement(root, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    #encoded_jpg_io = io.BytesIO(os.path.join(output_path, image_name))
    image = Image.open(os.path.join(output_path, image_name))
    width, height = image.size

    size = ET.SubElement(root, 'size')
    widthEle = ET.SubElement(size, 'width')
    widthEle.text = str(width)
    heightEle = ET.SubElement(size, 'height')
    heightEle.text = str(height)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    segmented = ET.SubElement(root, 'segmented')
    objectEle = ET.SubElement(root, 'object')
    object_name = ET.SubElement(objectEle, 'name')
    object_name.text = image_label
    object_pose = ET.SubElement(objectEle, 'pose')
    object_pose.text = 'Unspecified'
    object_truncated = ET.SubElement(objectEle, 'truncated')
    object_truncated.text = '0'
    object_difficult = ET.SubElement(objectEle, 'difficult')
    object_difficult.text = '0'
    object_bndbox = ET.SubElement(objectEle, 'bndbox')
    rois = args.roi.strip().split(',')
    xmin = ET.SubElement(object_bndbox, 'xmin')
    xmin.text = str(int(float(rois[0]) * width))
    ymin = ET.SubElement(object_bndbox, 'ymin')
    ymin.text = str(int(float(rois[1]) * height))
    xmax = ET.SubElement(object_bndbox, 'xmax')
    xmax.text = str(int((float(rois[2]) - float(rois[0])) * width))
    ymax = ET.SubElement(object_bndbox, 'ymax')
    ymax.text = str(int((float(rois[3]) - float(rois[1])) * height))

    xml_string = ET.tostring(root)
    xml_write = DOM.parseString(xml_string)
    xml_name = image_name.split('.')[0] + '.xml'

    with open(os.path.join(output_path, xml_name), 'w') as handle:
        xml_write.writexml(handle, indent='', addindent='\t', newl='\n', encoding='utf-8')

    print('[Generate]: ', os.path.join(output_path, xml_name))
    # 其中Rp是我构建的xml。

def bmpToJpg(input_path, fileName, output_path, output_name):
    #newFileName = os.path.basename(fileName) + ".jpg"
    image_np = cv2.imread(input_path + os.sep + fileName) # 用opencv 打开再存储，为了防止单通道图像的问题
    newpath = output_path + os.sep + output_name
    cv2.imwrite(newpath, image_np)
    return newpath

def generate_classify_xml(image_dir, output_path, label_map_dict):
    print(output_path)
    output_path = output_path.replace('\\', '/')
    #output_path = os.path.join(output_path, '/')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # enumerate each image, copy to image dir and generate xml which corresponded to the image
    for name in label_map_dict.keys():
        sub_dir = os.path.join(image_dir, name)
        for image in os.listdir(sub_dir):
            # .jpg
            gen_img_name = name + '_' + image.split('.')[0] + '.jpg'

            'if the image type is bmp, then convert to jpg'
            imageObj = Image.open(os.path.join(image_dir,sub_dir,image))
            if imageObj.format != 'JPEG':
                if imageObj.format == "BMP":
                    bmpToJpg(os.path.join(image_dir,sub_dir), image, output_path, gen_img_name)
            else:
                copyfile(os.path.join(image_dir,sub_dir,image), os.path.join(output_path, gen_img_name))

            generate_xml(gen_img_name, name, output_path)

if args.task_type == "classify":
    if args.labels_path is None:
        # generate label_map.pbtxt
        print("generate classify default label_map.pbtxt")
        args.labels_path = generate_classify_labels_file(args.image_dir)


print(args.labels_path)
label_map = label_map_util.load_labelmap(args.labels_path)
label_map_dict = label_map_util.get_label_map_dict(label_map)
print(label_map_dict)


def xml_to_csv(path):
    """Iterates through all .xml files (generated by labelImg) in a given directory and combines
    them in a single Pandas dataframe.

    Parameters:
    ----------
    path : str
        The path containing the .xml files
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """

    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def class_text_to_int(row_label):
    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    generate_classify_xml(args.image_dir, args.output_path, label_map_dict)
    return
    writer = tf.python_io.TFRecordWriter(args.output_path)
    path = os.path.join(args.image_dir)
    examples = xml_to_csv(args.xml_dir)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecord file: {}'.format(args.output_path))
    if args.csv_path is not None:
        examples.to_csv(args.csv_path, index=None)
        print('Successfully created the CSV file: {}'.format(args.csv_path))


if __name__ == '__main__':
    tf.app.run()
