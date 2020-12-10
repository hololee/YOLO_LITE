import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from models import config as cfg


def plot_box(img, bboxes, classes, confidence=cfg.VALID_OUTPUT_THRESHOLD):
    '''
    :param img: numpy array.
    :param bboxes: numpy array (b, 4)  # top_left_x, top_left_y, bottom_right_x, bottom_right_y
    :param classes: numpy array (b, n_class)
    :return:
    '''
    plt.imshow(img)
    H, W, _ = img.shape

    # Get axis.
    ax = plt.gca()

    for id, box in enumerate(bboxes):
        if max(classes[id]) > confidence:
            # Rectangle((left-top corner), width, height)
            if np.argmax(classes[id]) == 0:
                rect = patches.Rectangle((box[0], box[1]),
                                         box[2] - box[0],
                                         box[3] - box[1],
                                         linewidth=2,
                                         edgecolor='red',
                                         fill=False)
            else:
                rect = patches.Rectangle((box[0], box[1]),
                                         box[2] - box[0],
                                         box[3] - box[1],
                                         linewidth=2,
                                         edgecolor='blue',
                                         fill=False)
            ax.add_patch(rect)

    plt.show()
