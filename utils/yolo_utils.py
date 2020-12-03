import torch
from torchvision.transforms import functional as F
import models.config as cfg


# Change data to tensor type.
class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


# Compose all transforms.
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# collate_fn for different size of input image.
def collate_fn(batch):
    return tuple(zip(*batch))


def calculate_loss(gpu, output, target):
    '''
    :param gpu: device instance.
    :param output: yolo output feature. (N, C, H, W)
    :param target: ('N' tuple of dictionary) target information. ({"boxes": boxes, "classes": classes}, {"boxes": boxes, "classes": classes} ....)
                                        boxes_shape = [cell_x(index), cell_y(index), center_x center_y, w, h], classes_shape = [m]
     :param optimizer: yolo_optimizer
    :return:
    '''

    # Initialize loss.
    loss_bounding_box = None
    loss_object_conf = None
    loss_class_prob = None
    loss_noon_object_conf = None

    # target feature. (N, C, H, W)
    target_feature = torch.zeros(output.shape)
    # move to gpu.
    output.to(gpu)
    target_feature.to(gpu)

    # Rotate batch images.
    for idx, one_output in enumerate(output):
        one_target = target[idx]
        boxes, classes = one_target['boxes'], one_target['classes']
        # Rotate G.T bounding boxes.

        # all indices of cells.
        all_cell_indices = [[i, j] for i in range(cfg.grid) for j in range(cfg.grid)]

        # calculate loss for object responsible.
        for jdx in range(len(boxes)):
            # cell index. (x , y)
            index_cell = boxes[jdx][:2].detach().numpy()

            # remove from all_cell_indices for calculate on not responsible on non-objects.
            all_cell_indices.remove(index_cell)
            one_box_target = boxes[jdx][2:]  # (4)
            one_class_target = classes[jdx]
            class_prob = torch.zeros([cfg.n_classes])
            class_prob[one_class_target.data] = 1

            # move to gpu.
            one_box_target.to(gpu)

            # NOTICE: Calculate Bounding Box loss.

            # NOTICE: Calculate object confidence loss.

            # NOTICE: Calculate Class Probability loss.

        # NOTICE: calculate object confidence for non-object.
        for remain_non_object_indices in all_cell_indices:
            # one_output[:, remain_non_object_indices[1], remain_non_object_indices[0]]
            pass
    total_loss = (cfg.lambda_coord * loss_bounding_box) + loss_object_conf + (cfg.lambda_noobj * loss_noon_object_conf) + loss_class_prob
    return total_loss
