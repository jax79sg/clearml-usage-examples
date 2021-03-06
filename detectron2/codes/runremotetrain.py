from clearml import Task, Logger
task = Task.init(project_name='Balloon Detection',task_name='Detectron3',task_type='training', output_uri='s3://jax79sg.hopto.org:9000/clearml-models/')
task.set_base_docker("quay.io/jax79sg/detectron2:v4 --env GIT_SSL_NO_VERIFY=true --env TRAINS_AGENT_GIT_USER=testuser --env TRAINS_AGENT_GIT_PASS=testuser" )
task.execute_remotely(queue_name="1xV100-4ram", exit_process=True)


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import boto3
import argparse
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from botocore.client import Config

def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
                else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)

def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts







if __name__ == "__main__": 
   from clearml.config import config_obj
   print(config_obj.get('sdk.aws.s3.credentials'))
   parser = argparse.ArgumentParser()
   parser.add_argument('--TRAIN',type=str,default="balloon_train",required=False,help='Path to training dataset')
   parser.add_argument('--NUM_WORKERS',type=int,default=2,required=False,help='Number of workers for dataloading')
   parser.add_argument('--IMS_PER_BATCH',type=int,default=2,required=False,help='Training batch size')
   parser.add_argument('--BASE_LR',type=float,default=0.00025,required=False,help='Initial learning rate')
   parser.add_argument('--MAX_ITER',type=int,default=200,required=False,help='Maximum number of iterations')
   parser.add_argument('--BATCH_SIZE_PER_IMAGE',type=int,default=128,required=False,help='MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE')
   parser.add_argument('--NUM_CLASSES',type=int,default=1,required=False,help='Number of classes')
   args = parser.parse_args()
   print("Args",args)

   
   
### PULLING DATA FROM RELEVANT PLACES ###
   dataset_id = "2bc362f3c6194739bfe2b7bcf75b574a"
   datasets_used=dict(dataset_id="2bc362f3c6194739bfe2b7bcf75b574a")
   task.connect(datasets_used, name='datasets')
   from clearml import Dataset
   print("Downloading Datasets ... ",end="")
   dataset_path = Dataset.get(dataset_id=datasets_used['dataset_id']).get_local_copy()
   print("done")
   print(dataset_path)

   s3=boto3.resource('s3',
        endpoint_url='http://jax79sg.hopto.org:9000',
        aws_access_key_id='jax',
        aws_secret_access_key='P@ssw0rd',
        config=Config(signature_version='s3v4'),
        region_name='us-east-1',
        verify=False)

   print("Downloading pretrained models ... ",end="")
   s3.Bucket('clearml-models').download_file('detectron2/coco-instancesegmentation/mask_rcnn_R_50_FPN_3x/mask_rcnn_R_50_FPN_3x.yaml','/home/appuser/detectron2_repo/detectron2/model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
   s3.Bucket('clearml-models').download_file('detectron2/coco-instancesegmentation/mask_rcnn_R_50_FPN_3x/model_final_f10217.pkl','/home/appuser/model_final_f10217.pkl')
   print("Done")
 
### Registering dataset as per required by Detectron2 ###  
   print("Registering Datasets into Detectron ... ",end="")
   for d in ["train", "val"]:
      DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts(dataset_path+"/" + d))
      MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
   balloon_metadata = MetadataCatalog.get("balloon_train")
   print("Done")   
   

### Randomly sampling and showing dataset ###
   dataset_dicts = get_balloon_dicts(dataset_path+"/train")
   for d in random.sample(dataset_dicts, 3):
      img = cv2.imread(d["file_name"])
      visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
      out = visualizer.draw_dataset_dict(d)
      imout=out.get_image()[:, :, ::-1]
      cv2.imwrite("Sample_"+d["file_name"], imout)
      
### CONFIGURING THE MODEL ###
   cfg = get_cfg()
   cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
   print("Configuration from model zoo".format(cfg))
   cfg.DATASETS.TRAIN = (args.TRAIN)
   cfg.DATASETS.TEST = ()
   cfg.DATALOADER.NUM_WORKERS = args.NUM_WORKERS
   #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
   cfg.SOLVER.IMS_PER_BATCH = args.IMS_PER_BATCH
   cfg.SOLVER.BASE_LR = args.BASE_LR  # pick a good LR
   cfg.SOLVER.MAX_ITER = args.MAX_ITER    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
   cfg.SOLVER.STEPS = []        # do not decay learning rate
   cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.BATCH_SIZE_PER_IMAGE   # faster, and good enough for this toy dataset (default: 512)
   cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.NUM_CLASSES  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
   # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.      

### BEGINNING TRAINING ###  
   os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
   trainer = DefaultTrainer(cfg)
   trainer.resume_or_load(resume=False)   

### TRAINING THE MODEL ###
   #for i in range(0,args.MAX_ITER):
   # if (i%50==0):
   #   #Save inference every 50 iterations
   #   predictor = DefaultPredictor(cfg)
   #   outputs=predictor(im)
   #   v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
   #   out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
   #   imoutput=out.get_image()[:, :, ::-1]
   #   filename=str(i)+".jpg"
   #   cv2.imwrite(filename,imoutput)
   trainer.train()
   print("Training complete")

### INFERENCE SAMPLE ###
   im = cv2.imread("./pic.jpg")
   predictor = DefaultPredictor(cfg)
   outputs=predictor(im)
   v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
   out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
   imoutput=out.get_image()[:, :, ::-1]
   filename="pic-inferred.jpg"
   cv2.imwrite(filename,imoutput)
   print("Sample inference printed to ", filename)
   task.get_logger().report_image(title='Test Image', series='Original',iteration=1,image=im)
   task.get_logger().report_image(title='Test Image', series='Detection',iteration=1,image=imoutput)
