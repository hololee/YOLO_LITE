import os
import json
import csv
import torch
import numpy as np
from imageio import imread
from utils.yolo_utils import collate_fn, ToTensor, Compose
import models.config as cfg


class DataManager:
    '''
    NOTICE: <USING COCO DATASET FORMAT>
    '''

    def __init__(self, transforms, d_type):
        # Set transforms.
        self.transforms = transforms

        # Set dataset type.
        self.d_type = d_type

        # calculate cell size.
        self.cell_size_x = cfg.IMAGE_SIZE[1] / cfg.GRID
        self.cell_size_y = cfg.IMAGE_SIZE[0] / cfg.GRID

        # NOTICE: ########## parsing image info from annotations. ##########
        # Valid image names.
        self.names = set()

        # name2id : Dic, map name to image id.
        # id2name : Dic, map id to image name.
        self.name2id = {}
        self.id2name = {}

        # Set annotation.json file path.
        with open(cfg.ANNOTATION_PATH) as annotation_info:
            parsed_data = json.load(annotation_info)

        self.n_categories = len(parsed_data['categories'])

        # Make annotation file for each images.
        for data in parsed_data['images']:
            # Add information to name2id and id2name for efficient translations.
            self.name2id[data['file_name']] = data['id']
            self.id2name[data['id']] = data['file_name']

            # # Check annotated image is exist. (initializes)
            if os.path.exists(os.path.join(cfg.IMG_PATH, data['file_name'])):
                # Create annotation csv file for each image. and write file id.
                f = open(os.path.join(cfg.BOX_PATH, '{}.csv'.format(data['file_name'].split('.')[0])), 'w', encoding='utf-8')
                f.close()

        # Add box annotations to above generated file.
        for box_info in parsed_data['annotations']:
            # Check annotation file is generated.
            if os.path.exists(os.path.join(cfg.BOX_PATH, '{}.csv'.format(self.id2name[box_info['image_id']].split('.')[0]))):
                # write box info.
                f = open(os.path.join(cfg.BOX_PATH, '{}.csv'.format(self.id2name[box_info['image_id']].split('.')[0])), 'a', encoding='utf-8')
                wr = csv.writer(f)

                '''
                # Normalize the bounding box width and height by the image width and height so that they fall between 0 and 1.
                # Parametrize the bounding box x and y coordinates to be offsets of a particular grid cell location so they are also bounded between 0 and 1.
                # coco shape : class, [x-top-left, y-top-left, width, height]
                # yolo shape : [center_x, center_y, w, h] (ratio of image_size) in this code, add given index on grid,  [category, cell_x, cell_y, center_x center_y, w, h]
                # coordinate x is in width, coordinate y is in height.
                '''

                # change to ratio type
                gt_dat = box_info['bbox']

                # calculate center coordinates.
                center_x_raw = gt_dat[0] + (gt_dat[2] / 2)
                center_y_raw = gt_dat[1] + (gt_dat[3] / 2)

                # calculate cell index and ratio.
                cell_x = int((center_x_raw / cfg.IMAGE_SIZE[1]) * cfg.GRID)
                cell_y = int((center_y_raw / cfg.IMAGE_SIZE[0]) * cfg.GRID)
                center_x = ((center_x_raw / cfg.IMAGE_SIZE[1]) * cfg.GRID) - cell_x
                center_y = ((center_y_raw / cfg.IMAGE_SIZE[0]) * cfg.GRID) - cell_y

                # calculate w, h ratio.
                w = gt_dat[2] / cfg.IMAGE_SIZE[1]
                h = gt_dat[3] / cfg.IMAGE_SIZE[0]

                # write on csv file.
                wr.writerow([int(box_info['category_id']) - 1, cell_x, cell_y, center_x, center_y, w, h])

                f.close()

                # Add to Valid file name list.
                self.names.add(self.id2name[box_info['image_id']])

        # Change valid list from set to list.
        self.names = sorted(list(self.names))

        if self.d_type == 'train':
            self.names = self.names[:int(cfg.DATASET_DIVIDE_RATIO[0] * len(self.names))]

        elif self.d_type == 'dev':
            self.names = self.names[int(cfg.DATASET_DIVIDE_RATIO[0] * len(self.names)):int(sum(cfg.DATASET_DIVIDE_RATIO[:2]) * len(self.names))]
        else:
            self.names = self.names[int(sum(cfg.DATASET_DIVIDE_RATIO[:2]) * len(self.names)):]

        print(len(self.names), f'images {self.d_type} loaded.')

    def __getitem__(self, index):
        # Set each data path.
        file_img_path = os.path.join(cfg.IMG_PATH, self.names[index])
        file_boxes_path = os.path.join(cfg.BOX_PATH, '{}.csv'.format(self.names[index].split('.')[0]))

        # Load image.
        img = imread(file_img_path, pilmode='RGB')

        # Load boxes and corresponding class info.
        boxes = []
        classes = []

        # dat shape = yolo shape : [category, cell_x, cell_y, center_x center_y, w, h] (ratio of image_size)
        with open(file_boxes_path) as f:
            reader = csv.reader(f)

            # Read lines.
            for row in reader:
                # Set category and cell index to (int) , bbox to (float).
                dat = list(map(int, row[:3])) + list(map(float, row[3:]))
                boxes.append(dat[1:])
                classes.append(dat[0])

        # Generate target Dic.
        target = {"boxes": boxes, "classes": classes}

        # Apply transforms.
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.names)


def get_data_loader(d_type):
    # Prepare transforms.
    transforms = [ToTensor()]
    transforms = Compose(transforms)

    # Load Whole data.

    # Generate data_loader.
    data = DataManager(transforms, d_type)
    if d_type == 'train':
        data_loader = torch.utils.data.DataLoader(data,
                                                  batch_size=cfg.TRAINING_BATCH_SIZE,
                                                  shuffle=cfg.TRAINING_DATA_SHUFFLE,
                                                  num_workers=cfg.TRAINING_NUM_WORKERS,
                                                  collate_fn=collate_fn)
    elif d_type == 'dev':
        data_loader = torch.utils.data.DataLoader(data,
                                                  batch_size=cfg.TRAINING_BATCH_SIZE,
                                                  shuffle=cfg.TRAINING_DATA_SHUFFLE,
                                                  num_workers=cfg.TRAINING_NUM_WORKERS,
                                                  collate_fn=collate_fn)


    elif d_type == 'test':
        data_loader = torch.utils.data.DataLoader(data,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  collate_fn=collate_fn)

    return data_loader
