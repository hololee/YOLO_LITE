import torch
from data.DataManager import get_data_loader
from models.YololiteNet import yoloLite
from utils.yolo_utils import non_maximum_suppression, calculate_mAP
from utils.vision import plot_box
import numpy as np
from models import config as cfg

device = torch.device('cpu')
model = yoloLite(classes=cfg.N_CLASSES, bbox=cfg.N_BOXES)
model.load_state_dict(torch.load('/data_ssd3/LJH/pytorch_project/YOLO_LITE/weights/weights.pth', map_location=device))
model.eval()

test_loader = get_data_loader('test')

mAP_list = []
for iteration, (img, target) in enumerate(test_loader):
    inputs = torch.stack(img)
    outputs = model(inputs)

    # TEST: without NMS
    for id, output in enumerate(outputs):
        img = np.transpose(inputs[id].detach().numpy(), [1, 2, 0])
        # cbboxes, cconfidences = coordYOLO2CORNER(output)
        cbboxes, cconfidences = non_maximum_suppression(output)
        plot_box(img, cbboxes, cconfidences)

        one_mAP = calculate_mAP(cbboxes, cconfidences, target[id])
        mAP_list.append(one_mAP)

print(f'AP : {np.mean(mAP_list)}')
