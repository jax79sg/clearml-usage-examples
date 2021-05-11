from clearml import Task, Logger
task = Task.init(project_name='DETECTRON2',task_name='Default Model Architecture',task_type='training', output_uri='http://jax79sg.hopto.org:9000/clearml-models/artifact')
task.set_base_docker("quay.io/jax79sg/detectron2:v4 --env GIT_SSL_NO_VERIFY=true --env TRAINS_AGENT_GIT_USER=testuser --env TRAINS_AGENT_GIT_PASS=testuser" )
task.execute_remotely(queue_name="single_gpu", exit_process=True)


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


import boto3
from botocore.client import Config
s3=boto3.resource('s3',
        endpoint_url='http://jax79sg.hopto.org:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin',
        config=Config(signature_version='s3v4'),
        region_name='us-east-1',
        verify=False)


s3.Bucket('digitalhub').download_file('clearml-models/detectron2/coco-detection/fast_rcnn_R_50_FPN_1x/fast_rcnn_R_50_FPN_1x.yaml','/home/appuser/detectron2_repo/detectron2/model_zoo/configs/COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml')
s3.Bucket('digitalhub').download_file('clearml-models/detectron2/coco-detection/fast_rcnn_R_50_FPN_1x/model_final_e5f7ce.pkl','/home/appuser/model_final_e5f7ce.pkl')
print("Downnloaded models")

im = cv2.imread("./pic.jpg")
task.get_logger().report_image(title='Test Image', series='Raw',iteration=1,image=im)
#cv2.imshow(im)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml'))
print(cfg)
cfg.MODEL.DEVICE='cpu'
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.5
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml')
predictor = DefaultPredictor(cfg)
outputs=predictor(im)

print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
imoutput=out.get_image()[:, :, ::-1]
#cv2.imshow(im)
task.get_logger().report_image(title='Test Image', series='Inferred',iteration=1,image=imoutput)


