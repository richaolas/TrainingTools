import time
import tensorflow as tf
import numpy as np

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import cv2
import os
import random

physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0:

    for k in range(len(physical_devices)):

        tf.config.experimental.set_memory_growth(physical_devices[k], True)

        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))

    else:

        print("Not enough GPU hardware devices available")

# checkpoint dir

PATH_TO_CASE_BASE = '/media/dev/home/dev/TrainingTools/.data/officeflower'
PATH_TO_CHECKPOINT = PATH_TO_CASE_BASE + '/train'
PATH_TO_PIPELINE_CONFIG = PATH_TO_CASE_BASE + '/models/my_model_dir/pipeline.config'
PATH_TO_MODEL_DIR_TFLITE = PATH_TO_CASE_BASE + '/models/export_model_dir_tflite'
PATH_TO_SAVED_MODEL_TFLITE = PATH_TO_MODEL_DIR_TFLITE + "/saved_model"
# PATH_TO_OUTPUT_MODEL_TFLITE = PATH_TO_CASE_BASE + '/models/export_model_dir_tflite/model.tflite'
PATH_TO_OUTPUT_MODEL_TFLITE_QUANTIZED = PATH_TO_MODEL_DIR_TFLITE + '/model_quantized.tflite'
images_path = '/media/renjch/WORK/Samples/Customer/officeflower/JPEGImages'
tools_script_path = '/media/renjch/TECH/DL/models/research/object_detection/export_tflite_graph_tf2.py'

convert_saved_model_cmd = "{} --pipeline_config_path {} --trained_checkpoint_dir {} --output_directory{}".format(
    tools_script_path,
    PATH_TO_PIPELINE_CONFIG,
    PATH_TO_CHECKPOINT,
    PATH_TO_MODEL_DIR_TFLITE
)
print(convert_saved_model_cmd)


# /media/renjch/TECH/DL/models/research/object_detection/export_tflite_graph_tf2.py  \
#     --pipeline_config_path /media/dev/home/dev/TrainingTools/.data/officeflower/models/my_model_dir/pipeline.config \
#     --trained_checkpoint_dir /media/dev/home/dev/TrainingTools/.data/officeflower/train \
#     --output_directory /media/dev/home/dev/TrainingTools/.data/officeflower/models/export_model_dir_tflite


def file_name(file_dir):
    image_file = []
    for root, dirs, files in os.walk(file_dir):
        # print(root) #当前目录路径
        # print(dirs) #当前路径下所有子目录
        # print(files) #当前路径下所有非目录子文件
        for file in files:
            image_file.append(os.path.join(root, file))
    return image_file


def load_image_into_representative_data(path):
    width = 300
    height = 300
    img = Image.open(path)
    out = img.resize((width, height), Image.ANTIALIAS)
    image_np = np.array(out)

    # note 1/255.0， 这里我理解是 真实输入的数据，把 uint8 转换成真实输入的数据
    return np.expand_dims(image_np, 0).astype(np.float32) / 255.0


def representative_dataset():
    files = file_name(images_path)
    # [note] the size need be small than total sample count
    files = random.sample(files, 100)
    for file in files:
        yield [load_image_into_representative_data(file)]
        # yield [np.array(img[0], dtype=np.float32)] # also possible


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(PATH_TO_SAVED_MODEL_TFLITE)  # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# representative_dataset is required when specifying TFLITE_BUILTINS_INT8 or INT8 supported types.
converter.representative_dataset = representative_dataset
converter.inference_input_type = tf.uint8  # or tf.int8
# keep the output unchanged
# converter.inference_output_type = tf.uint8  # or tf.int8
tflite_model_quantized = converter.convert()

# Save the model.
with open(PATH_TO_OUTPUT_MODEL_TFLITE_QUANTIZED, 'wb') as f:
    f.write(tflite_model_quantized)

# val = os.system('ls -al')
# print(val)

converter_edgetpu_cmd = 'edgetpu_compiler -s -o {} {}'.format(PATH_TO_MODEL_DIR_TFLITE, PATH_TO_OUTPUT_MODEL_TFLITE_QUANTIZED)

os.system(converter_edgetpu_cmd)
