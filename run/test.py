import torch
from data.DataManager import get_data_loader
from models.YololiteNet import yoloLite
from utils.yolo_utils import coordYOLO2CORNER
from utils.vision import plot_box
import numpy as np
from models import config as cfg

device = torch.device('cpu')
model = yoloLite(classes=cfg.N_CLASSES, bbox=cfg.N_BOXES)
model.load_state_dict(torch.load('/data_ssd3/LJH/pytorch_project/YOLO_LITE/weights/weights.pth', map_location=device))
model.eval()

train_loader, dev_loader, test_loader = get_data_loader()

for iteration, (img, target) in enumerate(train_loader):
    inputs = torch.stack(img)
    outputs = model(inputs)

    # TEST: without NMS
    for id, output in enumerate(outputs):
        img = np.transpose(inputs[id].detach().numpy(), [1, 2, 0])
        cbboxes, cconfidences = coordYOLO2CORNER(output)
        plot_box(img, cbboxes, cconfidences, confidence=0.5)
