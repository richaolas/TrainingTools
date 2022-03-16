import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import sys

print(sys.path)

#
# def load_image_into_numpy_array(path):
#     """Load an image from file into a numpy array.
#
#     Puts image into numpy array to feed into tensorflow graph.
#     Note that by convention we put it into a numpy array with shape
#     (height, width, channels), where channels=3 for RGB.
#
#     Args:
#       path: the file path to the image
#
#     Returns:
#       uint8 numpy array with shape (img_height, img_width, 3)
#     """
#     img_data = tf.io.gfile.GFile(path, 'rb').read()
#     image = Image.open(BytesIO(img_data))
#     (im_width, im_height) = image.size
#     return np.array(image.getdata()).reshape(
#         (im_height, im_width, 3)).astype(np.uint8)
#
#
# def get_keypoint_tuples(eval_config):
#     """Return a tuple list of keypoint edges from the eval config.
#
#     Args:
#       eval_config: an eval config containing the keypoint edges
#
#     Returns:
#       a list of edge tuples, each in the format (start, end)
#     """
#     tuple_list = []
#     kp_list = eval_config.keypoint_edge
#     for edge in kp_list:
#         tuple_list.append((edge.start, edge.end))
#     return tuple_list
#
# # @title Choose the model to use, then evaluate the cell.
# MODELS = {'centernet_with_keypoints': 'centernet_hg104_512x512_kpts_coco17_tpu-32', 'centernet_without_keypoints': 'centernet_hg104_512x512_coco17_tpu-8'}
#
# model_display_name = 'centernet_with_keypoints' # @param ['centernet_with_keypoints', 'centernet_without_keypoints']
# model_name = MODELS[model_display_name]
#
# # Download the checkpoint and put it into models/research/object_detection/test_data/
#
# pipeline_config = os.path.join('models/research/object_detection/configs/tf2/',
#                                 model_name + '.config')
# model_dir = 'models/research/object_detection/test_data/checkpoint/'
#
# # Load pipeline config and build a detection model
# configs = config_util.get_configs_from_pipeline_file(pipeline_config)
# model_config = configs['model']
# detection_model = model_builder.build(
#       model_config=model_config, is_training=False)
#
# # Restore checkpoint
# ckpt = tf.compat.v2.train.Checkpoint(
#       model=detection_model)
# ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()
#
# def get_model_detection_function(model):
#   """Get a tf.function for detection."""
#
#   @tf.function
#   def detect_fn(image):
#     """Detect objects in image."""
#
#     image, shapes = model.preprocess(image)
#     prediction_dict = model.predict(image, shapes)
#     detections = model.postprocess(prediction_dict, shapes)
#
#     return detections, prediction_dict, tf.reshape(shapes, [-1])
#
#   return detect_fn
#
# detect_fn = get_model_detection_function(detection_model)