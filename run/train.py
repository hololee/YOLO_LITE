import os
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
# device = torch.device('cpu')
device = torch.device(cfg.TRAINING_GPU if torch.cuda.is_available() else 'cpu')

# Setting model.
yoloLite.train()
yoloLite.to(device)

# Set optimizer.
params = [p for p in yoloLite.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=cfg.LEARNING_RATE, weight_decay=cfg.weight_decay)

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
    fig, ax = plt.subplots(2, 1)
    ax[0].set_title('Train set loss', fontsize=13)
    ax[0].plot([i for i in range(len(train_loss))], train_loss, linestyle='-',
               dash_joinstyle='round')
    ax[0].grid(True, alpha=0.3)

    ax[1].set_title('Develop set loss', fontsize=13)
    ax[1].plot([i for i in range(len(dev_loss))], dev_loss, color='red', linestyle='-',
               dash_joinstyle='round')
    ax[1].grid(True, alpha=0.3)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    plt.savefig(os.path.join(cfg.OUTPUT_PATH, 'learning_graph.png'))
    plt.show()
    plt.clf()

# Save trained weight.
torch.save(yoloLite.state_dict(),
           '../weights/weights.pth')
