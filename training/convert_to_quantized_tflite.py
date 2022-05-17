import subprocess

PATH_TO_CHECKPOINT = 'C:/Users/renjch/source/repos/TrainingTools/.data/officeflower/train'
PATH_TO_PIPELINE_CONFIG = 'C:/Users/renjch/source/repos/TrainingTools/.data/officeflower/models/my_model_dir/pipeline.config'
PATH_TO_MODEL_DIR_TFLITE = 'C:/Users/renjch/source/repos/TrainingTools/.data/officeflower/models/export_model_dir_tflite'
PATH_TO_SAVED_MODEL_TFLITE = PATH_TO_MODEL_DIR_TFLITE + "/saved_model"
PATH_TO_OUTPUT_MODEL_TFLITE_QUANTIZED = 'C:/Users/renjch/source/repos/TrainingTools/.data/officeflower/models/export_model_dir_tflite/model_quantized2.tflite'

cmd = 'python D:/DL/models/research/object_detection/export_tflite_graph_tf2.py  \
    --pipeline_config_path {} \
    --trained_checkpoint_dir {} \
    --output_directory {}' \
      .format(PATH_TO_PIPELINE_CONFIG, PATH_TO_CHECKPOINT, PATH_TO_MODEL_DIR_TFLITE)

print(cmd)
