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

    # Draw lines and labels.
    for id, box in enumerate(bboxes):
        if max(classes[id]) > confidence:
            # Rectangle((left-top corner), width, height)
            rect = patches.Rectangle((box[0], box[1]),
                                     box[2] - box[0],
                                     box[3] - box[1],
                                     linewidth=2,
                                     edgecolor=cfg.color_map[np.argmax(classes[id])],
                                     fill=False)

            ax.add_patch(rect)
            plt.text(box[0], box[1],
                     cfg.CLASS_NAME[np.argmax(classes[id])] + ': ' + str(max(classes[id]))[:5],
                     fontsize=12,
                     fontweight='medium',
                     color='w',
                     verticalalignment='bottom',
                     horizontalalignment='left',
                     bbox=dict(facecolor=cfg.color_map[np.argmax(classes[id])],
                               linewidth=1.5,
                               alpha=1))

    plt.axis('off')
    plt.show()
