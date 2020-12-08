import torch
from data.DataManager import get_data_loader
from models import config as cfg
from models.YololiteNet import yoloLite as YL
from utils.yolo_utils import train_step, forward_step
import matplotlib.pyplot as plt

# Load data_loader.
train_loader = get_data_loader('train')
develop_loader = get_data_loader('dev')

# Load model and set to train state.
yoloLite = YL(classes=cfg.N_CLASSES, bbox=cfg.N_BOXES)

# allocate GPUs.
device = torch.device(cfg.TRAINING_GPU if torch.cuda.is_available() else 'cpu')

# Setting model.
yoloLite.train()
yoloLite.to(device)

# Set optimizer.
params = [p for p in yoloLite.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=cfg.LEARNING_RATE)

# for keep training loss.
train_loss = []
dev_loss = []

# NOTICE: TRAINING...
for epoch in range(cfg.TOTAL_EPOCH):
    print(f'EPOCH : {epoch + 1} / {cfg.TOTAL_EPOCH}')

    for iteration, (img, target) in enumerate(train_loader):
        print(f'iteration : {iteration + 1}')

        # train one step.
        train_loss = train_step(device, yoloLite, img, target, optimizer, train_loss)

    print('Calculate dev loss...')

    for iteration, (img, target) in enumerate(develop_loader):
        # dev one step.
        dev_loss = forward_step(device, yoloLite, img, target, dev_loss)

    # plot train loss.
    plt.subplot(2, 1, 1)
    plt.plot([i for i in range(len(train_loss))], train_loss)
    plt.subplot(2, 1, 2)
    plt.plot([i for i in range(len(dev_loss))], dev_loss)
    plt.show()

# Save trained weight.
torch.save(yoloLite.state_dict(), '/data_ssd3/LJH/pytorch_project/YOLO_LITE/weights/weights.pth')
