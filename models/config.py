'''
configuration file for training YOLO LITE model.
'''
# NOTICE : ########## Data Related ###########
image_size = (512, 512)  # (h, w)
n_classes = 2
img_path = '/data_ssd3/LJH/pytorch_project/YOLO_LITE/data/PennFudanPed/image/'
annotation_path = '/data_ssd3/LJH/pytorch_project/YOLO_LITE/data/PennFudanPed/annotation/annotation.json'
box_path = '/data_ssd3/LJH/pytorch_project/YOLO_LITE/data/PennFudanPed/boxes/'

# NOTICE : ########## Model Related ###########
grid = 7
n_boxes = 2

# NOTICE : ########## Train Related ###########
different_image_size = False
learning_rate = 0.05
dataset_divide_ratio = [0.8, 0.1, 0.1]
training_batch_size = 4
training_data_shuffle = True
training_num_workers = 0
training_gpu = 3
total_epoch = 30
lambda_coord = 5
lambda_noobj = 0.5
