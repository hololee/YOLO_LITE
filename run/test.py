import torch
from data.DataManager import get_data_loader
from models.YololiteNet import yoloLite
from utils.yolo_utils import non_maximum_suppression, calculate_confusion, calculate_AP
from utils.vision import plot_box
import numpy as np
from models import config as cfg

device = torch.device('cpu')
model = yoloLite(classes=cfg.N_CLASSES, bbox=cfg.N_BOXES)
model.load_state_dict(torch.load('/data_ssd3/LJH/pytorch_project/YOLO_LITE/weights/weights.pth', map_location=device))
model.eval()

test_loader = get_data_loader('test')

# class ap_list
ap_list = []

# class confidence list.
all_confidences = None

# TP, FP list.
all_con_list = None

# TP +FN, all boxes.
all_predict = 0

for iteration, (img, target) in enumerate(test_loader):
    inputs = torch.stack(img)
    outputs = model(inputs)

    # TEST: without NMS
    for id, output in enumerate(outputs):
        img = np.transpose(inputs[id].detach().numpy(), [1, 2, 0])
        # cbboxes, cconfidences = coordYOLO2CORNER(output)
        cbboxes, cconfidences = non_maximum_suppression(output)
        plot_box(img, cbboxes, cconfidences)

        cconfidences, con_list = calculate_confusion(cbboxes, cconfidences, target[id])

        all_confidences = cconfidences if all_confidences is None else np.concatenate([all_confidences, cconfidences], axis=0)
        all_con_list = con_list if all_con_list is None else np.concatenate([all_con_list, con_list], axis=0)

# sort all confidence.
classes = np.argmax(all_confidences, axis=1)
all_confidences = np.max(all_confidences, axis=1)

sorted_index = np.argsort(all_confidences, axis=0)[::-1]

# Sort confidence by descending order.
all_confidences = np.squeeze(all_confidences[sorted_index])
if cfg.N_CLASSES == 1: all_confidences = np.expand_dims(all_confidences, axis=-1)
all_con_list = np.squeeze(all_con_list[sorted_index])

# calculate TP +FN, all boxes.
all_predict = len(all_confidences)

for c in range(cfg.N_CLASSES):
    # Using 'all_predict', 'output_confidences' and 'con_type', calculate AP for each class.

    # choose specific class.
    one_class_confidence = all_confidences[np.where(classes == c)]
    one_class_con_list = all_con_list[np.where(classes == c)]

    # Precision and recall.
    pre_rec = np.zeros([len(one_class_confidence), 2])

    for row in range(len(one_class_confidence)):
        precision = sum(one_class_con_list[:row + 1]) / (row + 1)
        recall = sum(one_class_con_list[:row + 1]) / all_predict

        # Update pre_rec
        pre_rec[row] = [precision, recall]

    # Add to ap_list
    ap_list.append(calculate_AP(pre_rec, plot=True))
