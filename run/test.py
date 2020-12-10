import torch
from data.DataManager import get_data_loader
from models.YololiteNet import yoloLite
from utils.yolo_utils import non_maximum_suppression, mAPCalculator
from utils.vision import plot_box
import numpy as np
from models import config as cfg

device = torch.device('cpu')
model = yoloLite(classes=cfg.N_CLASSES, bbox=cfg.N_BOXES)
model.load_state_dict(torch.load('/data_ssd3/LJH/pytorch_project/YOLO_LITE/weights/weights.pth', map_location=device))
model.eval()

# load dataset.
test_loader = get_data_loader('test')

# mAP calculator class.
cal_mAP = mAPCalculator()

for iteration, (img, target) in enumerate(test_loader):
    inputs = torch.stack(img)
    outputs = model(inputs)

    # TEST: without NMS
    for id, output in enumerate(outputs):
        img = np.transpose(inputs[id].detach().numpy(), [1, 2, 0])

        # output boxes after NMS.
        # cbboxes shape : [n_output_boxes, 4]
        # cconfidences shape : [n_output_boxes, n_classes]
        cbboxes, cconfidences = non_maximum_suppression(output)

        # Plot predict boxes on image.
        plot_box(img, cbboxes, cconfidences)

        # Keep predict boxes information for calculate mAP.
        cal_mAP.keep(cbboxes, cconfidences, target[id])

# Calculate mAP.
result = cal_mAP.calculate(plot=True, mean=True)
print(f'mAP : {result}')
