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

    def __init__(self, transforms):
        # Set transforms.
        self.transforms = transforms

        # calculate cell size.
        self.cell_size_x = cfg.image_size[1] / cfg.grid
        self.cell_size_y = cfg.image_size[0] / cfg.grid

        # NOTICE: ########## parsing image info from annotations. ##########
        # Valid image names.
        self.names = set()

        # name2id : Dic, map name to image id.
        # id2name : Dic, map id to image name.
        self.name2id = {}
        self.id2name = {}

        # Set annotation.json file path.
        with open(cfg.annotation_path) as annotation_info:
            parsed_data = json.load(annotation_info)

        # Make annotation file for each images.
        for data in parsed_data['images']:
            # Add information to name2id and id2name for efficient translations.
            self.name2id[data['file_name']] = data['id']
            self.id2name[data['id']] = data['file_name']

            # Check annotated image is exist.
            if os.path.exists(os.path.join(cfg.img_path, data['file_name'])):
                # Create annotation csv file for each image. and write file id.
                f = open(os.path.join(cfg.box_path, '{}.csv'.format(data['file_name'].split('.')[0])), 'w', encoding='utf-8')
                wr = csv.writer(f)
                wr.writerow(data['id'])
                f.close()

        # Add box annotations to above generated file.
        for box_info in parsed_data['annotations']:
            # Check annotation file is generated.
            if os.path.exists(os.path.join(cfg.box_path, '{}.csv'.format(self.id2name[box_info['image_id']].split('.')[0]))):
                # write box info.
                f = open(os.path.join(cfg.box_path, '{}.csv'.format(self.id2name[box_info['image_id']].split('.')[0])), 'w', encoding='utf-8')
                wr = csv.writer(f)

                # NOTICE: Normalize the bounding box width and height by the image width and height so that they fall between 0 and 1.
                # NOTICE: parametrize the bounding box x and y coordinates to be offsets of a particular grid cell location so they are also bounded between 0 and 1.
                # coco shape : class x-top-left y-top-left width height
                # yolo shape : center_x center_y w, h (ratio of image_size) in this code, add given index on grid,  [category, cell_x, cell_y, center_x center_y w, h]
                # coordinate x is in width, coordinate y is in height.

                # change to ratio type
                gt_dat = np.fromstring(box_info['bbox'].replace('[', '').replace(']', ''), dtype=float, sep=',')

                # calculate center coordinates.
                center_x_raw = gt_dat[0] + (gt_dat[2] / 2)
                center_y_raw = gt_dat[1] + (gt_dat[3] / 2)

                # calculate cell index and ratio.
                cell_x = (center_x_raw // cfg.image_size[1] * cfg.grid)
                cell_y = (center_y_raw // cfg.image_size[0] * cfg.grid)
                center_x = (center_x_raw / cfg.image_size[1] * cfg.grid) - cell_x
                center_y = (center_y_raw / cfg.image_size[0] * cfg.grid) - cell_y

                # calculate w, h ratio.
                w = gt_dat[2] / cfg.image_size[1]
                h = gt_dat[3] / cfg.image_size[0]

                # write on csv file.
                wr.writerow([box_info['category_id'], cell_x, cell_y, center_x, center_y, w, h])
                f.close()

                # Add to Valid file name list.
                self.names.add(self.id2name[box_info['image_id']])

                # Change valid list from set to list.
                self.names = sorted(list(self.names))

    def __getitem__(self, index):
        # Set each data path.
        file_img_path = os.path.join(cfg.img_path, self.names[index])
        file_boxes_path = os.path.join(cfg.box_path, '{}.csv'.format(self.names[index].split('.')[0]))

        # Load image.
        img = imread(file_img_path, pilmode='RGB')

        # Load boxes and corresponding class info.
        boxes = []
        classes = []

        # dat shape = yolo shape : center_x center_y w, h (ratio of image_size)
        with open(file_boxes_path) as f:
            reader = csv.reader(f)
            # First line is image id.
            reader.next()
            for row in reader:
                dat = list(row)
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


def get_data_loader():
    # Prepare transforms.
    transforms = [ToTensor()]
    transforms = Compose(transforms)

    # Load Whole data.
    whole_data = DataManager(transforms)

    # Divide dataset.
    data_size = len(whole_data)
    train_size = int(data_size * cfg.dataset_divide_ratio[0])
    dev_size = (data_size - train_size) // 2

    indices = torch.randperm(len(whole_data)).tolist()
    train_data = torch.utils.data.Subset(whole_data, indices[:train_size])
    dev_data = torch.utils.data.Subset(whole_data, indices[train_size:train_size + dev_size])
    test_data = torch.utils.data.Subset(whole_data, indices[train_size + dev_size:])

    # Generate data_loader.
    data_loader_train = torch.utils.data.DataLoader(train_data,
                                                    batch_size=cfg.training_batch_size,
                                                    shuffle=cfg.training_data_shuffle,
                                                    num_workers=cfg.training_num_workers,
                                                    collate_fn=collate_fn)

    data_loader_dev = torch.utils.data.DataLoader(dev_data,
                                                  batch_size=cfg.training_batch_size,
                                                  shuffle=cfg.training_data_shuffle,
                                                  num_workers=cfg.training_num_workers,
                                                  collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(test_data,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   collate_fn=collate_fn)

    return data_loader_train, data_loader_dev, data_loader_test
