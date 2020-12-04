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
        # one output :  # (C, H, W)
        one_target = target[idx]
        boxes, classes = one_target['boxes'], one_target['classes']  # (n_box, 6), (n_box, 1)
        # Rotate G.T bounding boxes.

        # all indices of cells.
        all_cell_indices = [[i, j] for i in range(cfg.grid) for j in range(cfg.grid)]

        # calculate loss for object responsible.
        for jdx in range(len(boxes)):
            # cell index. (x , y)
            index_cell = boxes[jdx][:2].detach().numpy()

            # remove from all_cell_indices for calculate on not responsible on non-objects.
            all_cell_indices.remove(index_cell)

            # prepare target information.
            one_box_target = boxes[jdx][2:]  # (4,)
            one_class_target_raw = classes[jdx]  # (1,)
            one_class_target = torch.zeros([cfg.n_classes])  # (C,)
            one_class_target[one_class_target_raw.data] = 1  # one-hot encoding.

            # move to gpu.
            one_box_target.to(gpu)

            # calculate on multiple bounding box prediction.
            for n_box in range(cfg.n_boxes):
                # NOTICE: Calculate Bounding Box loss.

                # center_x
                loss_bounding_box += torch.square(torch.sub(one_box_target[0], one_output[(5 * n_box), index_cell[1], index_cell[0]]))

                # center_y
                loss_bounding_box += torch.square(torch.sub(one_box_target[1], one_output[(5 * n_box) + 1, index_cell[1], index_cell[0]]))

                # width
                loss_bounding_box += torch.square(torch.sub(torch.sqrt(one_box_target[2]), torch.sqrt(one_output[(5 * n_box) + 2, index_cell[1], index_cell[0]])))

                # height
                loss_bounding_box += torch.square(torch.sub(torch.sqrt(one_box_target[3]), torch.sqrt(one_output[(5 * n_box) + 3, index_cell[1], index_cell[0]])))

                # NOTICE: Calculate object confidence loss.
                loss_object_conf += torch.square(torch.sub(1, one_output[(5 * n_box) + 4, index_cell[1], index_cell[0]]))  # target is 1.

            # NOTICE: Calculate Class Probability loss.
            loss_class_prob += torch.square(torch.sub(one_class_target, one_output[(5 * cfg.n_boxes):, index_cell[1], index_cell[0]]))

        # calculate on multiple bounding box prediction.
        for n_box in range(cfg.n_boxes):
            # for non-object cells.
            for remain_non_object_indices in all_cell_indices:
                # NOTICE: calculate object confidence loss for non-object.
                loss_noon_object_conf += torch.square(torch.sub(0, one_output[(5 * n_box) + 4, remain_non_object_indices[1], remain_non_object_indices[0]]))  # target is 0.

    total_loss = (cfg.lambda_coord * loss_bounding_box) + loss_object_conf + (cfg.lambda_noobj * loss_noon_object_conf) + loss_class_prob
    return total_loss
