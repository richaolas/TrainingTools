```
python object_detection/model_main_tf2.py --pipeline_config_path=/media/dev/home/dev/TrainingTools/.data/officeflower/models/my_model_dir/pipeline.config --model_dir=/media/dev/home/dev/TrainingTools/.data/officeflower/train --num_train_steps=100 --sample_1_of_n_eval_examples=20 --alsologtostderr
```

#### 
1. train the model
2. export to saved_model then test
3. export to tflite_saved_model then convert saved_model to tflite 
4. export to tflite_saved_model then convert saved_model to qu-tflite 
5. compile the qu-tflite to edge-tpu, then deploy on the camera
6. 