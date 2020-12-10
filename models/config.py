'''
configuration file for training YOLO LITE model.
'''
# NOTICE : ########## Data Related ###########
IMAGE_SIZE = (512, 512)  # (h, w)
N_CLASSES = 1
CLASS_NAME = ['Bean']
IMG_PATH = '/data_ssd3/LJH/pytorch_project/YOLO_LITE/data/bean_leaf_noweed/images/'
ANNOTATION_PATH = '/data_ssd3/LJH/pytorch_project/YOLO_LITE/data/bean_leaf_noweed/annotation/test_coco.json'
BOX_PATH = '/data_ssd3/LJH/pytorch_project/YOLO_LITE/data/bean_leaf_noweed/boxes/'

# img_path = '/data_ssd3/LJH/pytorch_project/YOLO_LITE/data/coco/images'
# annotation_path = '/data_ssd3/LJH/pytorch_project/YOLO_LITE/data/coco/annotation/instances_train2014.json'
# box_path = '/data_ssd3/LJH/pytorch_project/YOLO_LITE/data/coco/boxes'

# NOTICE : ########## Model Related ###########
GRID = 8
N_BOXES = 1

# NOTICE : ########## Train Related ###########
DIFFERENT_IMAGE_SIZE = False
LEARNING_RATE = 0.0001
DATASET_DIVIDE_RATIO = [0.8, 0.1, 0.1]
TRAINING_BATCH_SIZE = 16
TRAINING_DATA_SHUFFLE = True
TRAINING_NUM_WORKERS = 0
TRAINING_GPU = 3
TOTAL_EPOCH = 300
LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5

# NOTICE: ########## Post Processing ##########
VALID_OUTPUT_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5

# NOTICE : ######### Plotting ##########
color_map = ['r', 'g', 'c', 'm', 'y', 'k', ]
