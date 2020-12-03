import torch
from data.DataManager import get_data_loader
from models import config as cfg
from models.YololiteNet import yoloLite as YL
from utils.yolo_utils import calculate_loss

# Load data_loader.
train_loader, dev_loader, test_loader = get_data_loader()

# Load model and set to train state.
yoloLite = YL(classes=1, bbox=2)
yoloLite.train()

# allocate GPUs.
device = torch.device(cfg.training_gpu if torch.cuda.is_available() else 'cpu')

# Set optimizer.
params = [p for p in yoloLite.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)


# One step for training.
def train_step(gpu, model, input, target, optimizer):
    # NOTICE: Calculate loss and train one step.

    ############################################## for different size input, not yolo case.##########################################
    # # calculate forward pass.
    # if cfg.different_image_size:
    #     # if input image size is different, calculate output separately. and then stack all output features(output shape is same).
    #     output = []
    #     for one_batch in img:
    #         output.append(model(one_batch))
    #         torch.stack(output)
    #     # image size is different.
    #     pass
    # else:
    #     output = model(torch.stack(img))
    #     # image size is same.
    #     pass
    ############################################## for different size input, not yolo case.##########################################

    output = model(torch.stack(img))
    return calculate_loss(gpu, output, target, optimizer)


# NOTICE: TRAINING...
for epoch in range(cfg.total_epoch):
    for iteration, (img, target) in enumerate(train_loader):
        loss = train_step(device, yoloLite, img, target, optimizer)
