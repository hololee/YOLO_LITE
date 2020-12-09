import torch
from torchvision.transforms import functional as F
import models.config as cfg
import numpy as np


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


# Change YOLO output to left_top, right_bottom format for easily calculate IOU, etc.
def coordYOLO2CORNER(output):
    # output shape : [(5 * bbox + n_class), grid_h, grid_w]
    # converted bboxes. shape : [B, [left_top_x, left_top_y, right_bottom_x, right_bottom_y]] (axis on, w, h, w, h)
    cbboxes = []

    # class confidence : shape : [B, [0, 0, ..1, 0]] (B, C)
    cconfidences = []

    # Get information of grid.
    grid = cfg.GRID
    n_box = cfg.N_BOXES
    origin_h, origin_w = cfg.IMAGE_SIZE

    # Calculate x index : Width
    for x in range(grid):
        # Calculate y index : Height
        for y in range(grid):
            c = output[:, y, x]  # object, class confidences.
            for box in range(n_box):
                # Get one predict coordinate.
                yolo_coord = c[box * 5: (box * 5) + 4]
                yolo_coord = yolo_coord.detach().numpy()

                # Convert to real size.
                yolo_coord[0] = (origin_w * x / grid) + (origin_w * yolo_coord[0] / grid)  # centerX
                yolo_coord[1] = (origin_h * y / grid) + (origin_h * yolo_coord[1] / grid)  # centerY
                yolo_coord[2] = yolo_coord[2] * origin_w  # Width
                yolo_coord[3] = yolo_coord[3] * origin_h  # Height

                #  Convert to bbox shape.
                c_bbox = np.array([yolo_coord[0] - (yolo_coord[2] / 2),  # left-top x
                                   yolo_coord[1] - (yolo_coord[3] / 2),  # left_top y
                                   yolo_coord[0] + (yolo_coord[2] / 2),  # right_bot x
                                   yolo_coord[1] + (yolo_coord[3] / 2)])  # right_bot y
                c_confidence = c[5 * n_box:] * c[(box * 5) + 4]
                # print(C[(box * 5) + 4].item())
                c_confidence = c_confidence.detach().numpy()

                cbboxes.append(c_bbox)
                cconfidences.append(c_confidence)

    # shape : [b, 4], [b, n_class]
    return cbboxes, cconfidences


def calculate_loss(gpu, output, target):
    '''
    :param gpu: device instance.
    :param output: yolo output feature. [N, C, H, W]
    :param target: ('N' tuple of dictionary) target information. ({"boxes": boxes, "classes": classes}, {"boxes": boxes, "classes": classes} ....)
                                        boxes_shape = [cell_x(index), cell_y(index), center_x center_y, w, h], classes_shape = [m]
     :param optimizer: yolo_optimizer
    :return:
    '''

    # Initialize loss.
    loss_bounding_box = torch.tensor(0, dtype=torch.float32).to(gpu)
    loss_object_conf = torch.tensor(0, dtype=torch.float32).to(gpu)
    loss_class_prob = torch.tensor(0, dtype=torch.float32).to(gpu)
    loss_noon_object_conf = torch.tensor(0, dtype=torch.float32).to(gpu)

    # Batch size
    b_n = output.shape[0]

    # Rotate batch images.
    for idx, one_output in enumerate(output):
        # one output :  # [C, H, W]
        one_target = target[idx]
        boxes, classes = one_target['boxes'], one_target['classes']  # [n_box, 6], [n_box, 1]
        # Rotate G.T bounding boxes.

        # All indices of cells.
        all_cell_indices = [[i, j] for i in range(cfg.GRID) for j in range(cfg.GRID)]

        # calculate loss for object responsible.
        for jdx in range(len(boxes)):
            # cell index. [x , y]
            index_cell = boxes[jdx][:2]

            # Some case, multiple object's center can be in same grid cell. so just using first one.
            if index_cell in all_cell_indices:

                # Remove from all_cell_indices for calculate on not  responsible on non-objects.
                all_cell_indices.remove(index_cell)

                # Prepare target information.
                one_box_target = torch.FloatTensor(boxes[jdx][2:])  # [4,]
                one_class_target = torch.zeros([cfg.N_CLASSES])  # [C,]
                if cfg.N_CLASSES > 1:
                    one_class_target_raw = classes[jdx]  # [1,]
                    one_class_target[one_class_target_raw] = 1  # one-hot encoding.
                else:
                    one_class_target[0] = 1

                    # Move to gpu.
                one_box_target = one_box_target.to(gpu)
                one_class_target = one_class_target.to(gpu)

                # Calculate on multiple bounding box prediction.
                for n_box in range(cfg.N_BOXES):
                    # NOTICE: Calculate Bounding Box loss.
                    # print('n_box : ', n_box)

                    # center_x
                    loss_bounding_box += torch.square((one_box_target[0] - one_output[(5 * n_box), index_cell[1], index_cell[0]]))

                    # center_y
                    loss_bounding_box += torch.square((one_box_target[1] - one_output[(5 * n_box) + 1, index_cell[1], index_cell[0]]))

                    # width
                    loss_bounding_box += torch.square((torch.sqrt(one_box_target[2]) - torch.sqrt(one_output[(5 * n_box) + 2, index_cell[1], index_cell[0]])))

                    # height
                    loss_bounding_box += torch.square((torch.sqrt(one_box_target[3]) - torch.sqrt(one_output[(5 * n_box) + 3, index_cell[1], index_cell[0]])))

                    # NOTICE: Calculate object confidence loss.
                    loss_object_conf += torch.square((1 - one_output[(5 * n_box) + 4, index_cell[1], index_cell[0]]))  # target is 1.

                # NOTICE: Calculate Class Probability loss.
                loss_class_prob += torch.sum(torch.square((one_class_target - one_output[(5 * cfg.N_BOXES):, index_cell[1], index_cell[0]])))

        # For non-object cells.
        for remain_non_object_indices in all_cell_indices:

            # Calculate on multiple bounding box prediction.
            for n_box in range(cfg.N_BOXES):
                # NOTICE: calculate object confidence loss for non-object.
                loss_noon_object_conf += torch.square((0 - one_output[(5 * n_box) + 4, remain_non_object_indices[1], remain_non_object_indices[0]]))  # target is 0.

    # Calculate total loss.
    total_loss = ((cfg.LAMBDA_COORD * loss_bounding_box) + loss_object_conf + (cfg.LAMBDA_NOOBJ * loss_noon_object_conf) + loss_class_prob) / b_n
    return total_loss


# One step for training.
def train_step(device, model, input, target, optimizer, train_loss):
    '''
    :param device: torch device.
    :param model: torch model.
    :param input: torch.Tensor(torch.cuda.FloatTensor)
    :param target: torch.Tensor(torch.cuda.FloatTensor)
    :param optimizer: torch optimizer
    :param train_loss:  loss list by step
    :return: train_loss
    '''
    # NOTICE: Calculate loss and train one step.

    # Forward-prop
    input_device = torch.stack(input).to(device)
    output = model(input_device)

    # Calculate loss.
    total_loss = calculate_loss(device, output, target)

    # Back-prop
    total_loss.backward()
    optimizer.step()

    # Keep and print train loss.
    train_loss.append(total_loss.detach().cpu().item())
    print(train_loss[-1])

    return train_loss


# one step forward.
def forward_step(device, model, input, target, step_loss):
    # Calculate loss and forward one step.

    '''
    :param device: torch device.
    :param model: torch model.
    :param input: torch.Tensor(torch.cuda.FloatTensor)
    :param target: torch.Tensor(torch.cuda.FloatTensor)
    :param step_loss: loss list by step.
    :return: step_loss
    '''

    # forward-prop
    input_device = torch.stack(input).to(device)
    output = model(input_device)

    with torch.no_grad():
        # Calculate loss.
        total_loss = calculate_loss(device, output, target).detach().cpu().item()
        # Keep and print train loss.
        step_loss.append(total_loss)
        print(step_loss[-1])

    return step_loss


def calculate_iou_matrix(*coordinates):
    # Divide to each coordinates.
    x11, y11, x12, y12 = coordinates[:4]
    x21, y21, x22, y22 = coordinates[4:]

    # Choose edge of intersection.
    x_1 = np.maximum(x11, x21)  # intersection Left-Top x
    y_1 = np.maximum(y11, y21)  # intersection Left-Top y
    x_2 = np.minimum(x12, x22)  # intersection Right-Bottom x
    y_2 = np.minimum(y12, y22)  # intersection Right-Bottom y

    # Calculate intersection area.
    inter_area = np.maximum((x_2 - x_1), 0) * np.maximum((y_2 - y_1), 0)

    # Calculate iou.
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)
    iou = inter_area / (box1_area + box2_area - inter_area)

    return iou


def calculate_iou_relation_matrix(cbboxes):
    # See below.
    repeat = np.repeat(cbboxes, len(cbboxes), axis=0)
    all_iou = calculate_iou_matrix(repeat[:, 0], repeat[:, 1], repeat[:, 2], repeat[:, 3],
                                   np.tile(cbboxes[:, 0], len(cbboxes)), np.tile(cbboxes[:, 1], len(cbboxes)), np.tile(cbboxes[:, 2], len(cbboxes)), np.tile(cbboxes[:, 3], len(cbboxes)))
    '''
    if n_boxes is 5,
    -----------------------------------------------------------------------------------------
    cbboxes = 
    [[7, 0, 4, 0],
     [4, 5, 0, 0],
     [3, 3, 1, 3],
     [8, 4, 2, 3],
     [1, 1, 3, 0]]
    -----------------------------------------------------------------------------------------
    
    *LT_x is,
    -----------------------------------------------------------------------------------------
    repeat = [7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1]
    tile = [7, 4, 3, 8, 1, 7, 4, 3, 8, 1, 7, 4, 3, 8, 1, 7, 4, 3, 8, 1, 7, 4, 3, 8, 1]
    index = [(0, 0), (0, 1)... (1, 0), (1, 1), (1, 2)...   (B,B)]
  -----------------------------------------------------------------------------------------
     
    * all_iou is correlation matrix of boxes,
    -----------------------------------------------------------------------------------------
      0 1 2 3 4 5 6 7
    0 x x x x x x x x
    1 x x x x x x x x
    2 x x x x x x x x
    3 . . . .  [iou between axis x, and y]
    4 . . .
    5 . .
    6 .
    7 .
    -----------------------------------------------------------------------------------------
    
    * reshape to [len(cbboxes), len(cbboxes)],
    -----------------------------------------------------------------------------------------
    all_iou = 
    [[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)...]
     [(1, 1), (1, 1), (1, 2), (1, 3), (1, 4)...]
     ....
     ...
     ..                                  (B, B)]]
    -----------------------------------------------------------------------------------------
    '''
    return all_iou.reshape([len(cbboxes), len(cbboxes)])


def non_maximum_suppression(output):
    # cbboxes : shape:(B, [top_left_x, top_left_y, bottom_right_x, bottom_right_y]) (b, 4)
    # cconfidences : shape:(B, [0, 0, ..1, 0]) (B, C)
    cbboxes, cconfidences = coordYOLO2CORNER(output)

    # Change to numpy array.
    cbboxes = np.array(cbboxes)
    cconfidences = np.array(cconfidences)

    # Remove low confidence.
    cconfidences[cconfidences < cfg.VALID_OUTPUT_THRESHOLD] = 0
    sorted_index = np.argsort(cconfidences, axis=0)[::-1]

    # Sort by large confidence.
    cconfidences = np.squeeze(cconfidences[sorted_index])
    if cfg.N_CLASSES == 1: cconfidences = np.expand_dims(cconfidences, axis=-1)
    cbboxes = np.squeeze(cbboxes[sorted_index])

    # Calculate iou matrix.
    iou_matrix = calculate_iou_relation_matrix(cbboxes)

    # Remove overlap box.
    for c in range(cfg.N_CLASSES):
        for idx_base, base_conf in enumerate(cconfidences[:, c]):
            for idx_cur, cur_conf in enumerate(cconfidences[:, c]):
                # only valid confidence.
                if idx_base != idx_cur and base_conf > 0:
                    # base and current iou is larger than NMS TH, remove.
                    if iou_matrix[idx_base, idx_cur] > cfg.NMS_IOU_THRESHOLD:
                        cconfidences[idx_cur, c] = 0

    # Remove not valid boxes from two list.
    valid = np.max(cconfidences, axis=1) > cfg.VALID_OUTPUT_THRESHOLD

    # Remove zero confidence boxes.
    cconfidences = cconfidences[valid]
    cbboxes = cbboxes[valid]

    return cbboxes, cconfidences
