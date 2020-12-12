import os
import torch
from torchvision.transforms import functional as F
import models.config as cfg
import numpy as np
import matplotlib.pyplot as plt


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
def get_output_boxes(output):
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


def YOLO2CORNER(yolo_coord):
    '''
    Change Yolo format to Corner format.
    from : [cell_x, cell_y, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    to : [Left-Top x, Left-Top y, Right-Bottom x, Right-Bottom y]
    '''

    # Get information of grid.
    grid = cfg.GRID
    origin_h, origin_w = cfg.IMAGE_SIZE

    # keep real size.
    converted_to_real = []

    # Convert to real size.
    converted_to_real.append((origin_w * yolo_coord[0] / grid) + (origin_w * yolo_coord[2] / grid))  # centerX
    converted_to_real.append((origin_h * yolo_coord[1] / grid) + (origin_h * yolo_coord[3] / grid))  # centerY
    converted_to_real.append(yolo_coord[4] * origin_w)  # Width
    converted_to_real.append(yolo_coord[5] * origin_h)  # Height

    #  Convert to bbox shape.
    c_bbox = np.array([converted_to_real[0] - (converted_to_real[2] / 2),  # left-top x
                       converted_to_real[1] - (converted_to_real[3] / 2),  # left_top y
                       converted_to_real[0] + (converted_to_real[2] / 2),  # right_bot x
                       converted_to_real[1] + (converted_to_real[3] / 2)])  # right_bot y

    return c_bbox


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
                    # just one class.
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
    iou = inter_area / (box1_area + box2_area - inter_area + 0.001)

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
    # cbboxes : shape:(n_bboxes, [top_left_x, top_left_y, bottom_right_x, bottom_right_y]) (b, 4)
    # cconfidences : shape:(n_bboxes, [0, 0, ..1, 0]) (B, C)
    cbboxes, cconfidences = get_output_boxes(output)

    # Change to numpy array.
    cbboxes = np.array(cbboxes)
    cconfidences = np.array(cconfidences)

    # Remove low confidence.
    cconfidences[cconfidences < cfg.VALID_OUTPUT_THRESHOLD] = 0

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

    return cbboxes, cconfidences


class mAPCalculator():
    def __init__(self):
        # class ap_list shape : (n_classes,)
        self.ap_list = []

        # class confidence list. shape : (all_predictions, n_classes)
        self.all_confidences = None

        # TP, FP list. shape : (all_predictions,) if iou with G.T > threshold, and class is correct, 1 else 0.
        self.all_con_list = None

    def keep(self, cbboxes, cconfidences, target_id):
        '''
        NOTICE: calculate prediction is TP or FP and keep prediction information.
        :param cbboxes: (n_output_boxes, 4)
        :param cconfidences: (n_output_boxes, n_classes)
        :param target_id: {"boxes": boxes, "classes": classes} boxes : [n_target_boxes, [cell_x, cell_y, center_x center_y, w, h]],  classes: [n_target_boxes, category]
        :return:
        '''

        '''
        1. First, collect all box class, confidence, confusion information. 
           (Just calculate precision recall for predictions on all Test Image.)
        -----------------------------------------------------------------------------------------------
                file        | class(argmax(confidences)) | confidence (:max(confidences)) | confusion
            image01.png                 2                           0.88951                   1 (TP)
            image02.png                 1                           0.93215                   1 (TP)
            image01.png                 0                           0.85331                   0 (FP) 
            image01.png                 0                           0.98245                   1 (TP)
            image02.png                 2                           0.90457                   0 (FP)
                   .
                   .
                   .
        -----------------------------------------------------------------------------------------------        
        NOTICE: divide data by class and calculate confusion and merge them again.
        NOTICE: when calculate confusion, if duplicated prediction occurred, set 0 confusion to next predictions.  
        '''

        # Calculate which prediction is TP or FP
        cconfidences, con_list = self.calculate_confusion(cbboxes, cconfidences, target_id)

        # Stack confidences of predictions and is TP or FP
        self.all_confidences = cconfidences if self.all_confidences is None else np.concatenate([self.all_confidences, cconfidences], axis=0)
        self.all_con_list = con_list if self.all_con_list is None else np.concatenate([self.all_con_list, con_list], axis=0)

    def calculate(self, plot=True, mean=True):
        '''
        :param plot:
        :param mean:
        :return:
        '''
        '''
        2. Sort by descending confidence.
           (Don't need to know about which predictions are placed on which image because just calculate precision recall for all predictions.)
        -------------------------------------------------------------------------------
        class(argmax(confidences)) | confidence (:max(confidences)) | confusion
                   0                           0.98245                   1 (TP)
                   1                           0.93215    |              1 (TP)
                   2                           0.90457    |              0 (FP) 
                   2                           0.88951    V              1 (TP)
                   0                           0.85331                   0 (FP)
                   .
                   .
                   .
        -------------------------------------------------------------------------------            
        
        3. divide matrix by class and calculate precision and recall.
        -------------------------------------------------------------------------------
        
        =================================class0================================
        class(argmax(confidences)) | confidence (:max(confidences)) | confusion
                0                           0.98245                   1 (TP)
                0                           0.85331                   0 (FP)
                .
                .
                .

        =================================class1================================
        class(argmax(confidences)) | confidence (:max(confidences)) | confusion
                1                           0.93215                   1 (TP)
                .
                .
                .

        =================================class2================================
        class(argmax(confidences)) | confidence (:max(confidences)) | confusion
                2                           0.90457                   0 (FP) 
                2                           0.88951                   1 (TP)
                .
                .
                .

        -------------------------------------------------------------------------------
        
        4. Calculate mAP or just show AP of each class.   
        '''

        # sort all confidence.
        classes = np.argmax(self.all_confidences, axis=1)
        all_confidences = np.max(self.all_confidences, axis=1)

        # Get sorted index by  descending class confidence
        sorted_index = np.argsort(all_confidences, axis=0)[::-1]

        # Sort confidence by descending order.
        all_confidences = np.squeeze(all_confidences[sorted_index])
        if cfg.N_CLASSES == 1: all_confidences = np.expand_dims(all_confidences, axis=-1)
        all_con_list = np.squeeze(self.all_con_list[sorted_index])

        for c in range(cfg.N_CLASSES):
            # Using 'all_predict', 'output_confidences' and 'con_type', calculate AP for each class.

            # choose specific class.
            one_class_confidence = all_confidences[np.where(classes == c)]
            one_class_con_list = all_con_list[np.where(classes == c)]

            # calculate TP +FN, all box predictions for in each class.
            all_predict = sum(one_class_con_list)

            # Precision and recall.
            pre_rec = np.zeros([len(one_class_confidence), 2])

            # Calculate precision and recall.
            for row in range(len(one_class_confidence)):
                precision = sum(one_class_con_list[:row + 1]) / (row + 1)
                recall = sum(one_class_con_list[:row + 1]) / all_predict

                # Update pre_rec
                pre_rec[row] = [precision, recall]

            # Add to ap_list (AP for each class.) shape:[C]
            ap = self.calculate_AP(c, pre_rec, plot)
            print(f'AP_{cfg.CLASS_NAME[c]} : {ap}')
            self.ap_list.append(ap)

        return np.mean(self.ap_list) if mean else self.ap_list

    def calculate_confusion(self, cbboxes, cconfidences, target):
        '''
        :param cbboxes: shape: (n_output_boxes, 4) ;[top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        :param cconfidences: shape:(n_output_boxes, n_classes)
        :param target:  {"boxes": boxes, "classes": classes} boxes : [n_target_boxes, [cell_x, cell_y, center_x center_y, w, h]],  classes: [n_target_boxes, category]
        :param plot: plotting or not
        :return:

        cconfidences: valid prediction boxes. shape: (n_valid_predictions, 1); ex) 0.73456
        con_list: valid prediction boxes is correct or not. shape: (n_valid_predictions, 1); 1 is TP, 0 is FP.
        '''

        # Remove not valid boxes from two list.
        valid = np.max(cconfidences, axis=1) > 0

        # Remove zero confidence boxes.
        cconfidences = cconfidences[valid]
        cbboxes = cbboxes[valid]

        # Get classes.
        classes = np.argmax(cconfidences, axis=1)

        # TP, FP list.  # TP or FP : TP is 1, FP is 0
        con_list = np.zeros([len(cbboxes), 1], dtype=int)

        # Calculate AP for each class.
        for c in range(cfg.N_CLASSES):

            # Get each class boxes.
            output_boxes = cbboxes[np.where(classes == c)]

            # Target boxes flag. It is for prevent duplicated prediction.
            target_flag = np.zeros([len(np.where(np.array(target['classes']) == c)[0]), ], dtype=int)

            # target_boxes(G.T bounding boxes) on specific class.
            target_boxes = np.array(target['boxes'])[np.where(np.array(target['classes']) == c)]

            # Set TP, FP
            for tdx, t_box in enumerate(target_boxes):

                # Compare with 'output_boxes' and calculate IOU. next, set TP or FP
                for bdx, output_box in enumerate(output_boxes):

                    # each box shape : top_left_x, top_left_y, bottom_right_x, bottom_right_y
                    iou = calculate_iou_matrix(*output_box, *YOLO2CORNER(t_box))
                    # print(f'iou : {iou}')

                    # If TP, (generally, iou >= 0.5)
                    if iou > cfg.VALID_OUTPUT_THRESHOLD and target_flag[tdx] == 0:
                        # Set to TP.
                        con_list[np.where(classes == c)[0][bdx]] = [1]

                        # Set flag to 1. (when calculate AP, duplicated detection at last come to not correct prediction.)
                        target_flag[tdx] = 1

        return cconfidences, con_list

    def calculate_AP(self, class_id, pre_rec, plot=False):
        '''
        :param class_id: class id.
        :param pre_rec: shape : (n_prediction_boxes, 2); [precision, recall]
        :return: AP
        '''

        '''
        |
        |---------;
        |         |
        |         |
        |         | 
        |    A    |
        |         ----------;
        |         .    B    |
        |         .         ----------;      
        |         .         .    C       ...
        |_______________________________________
                  |         |
                  T1        T2    ...
        '''

        # Plot precision recall graph.
        if plot:
            plt.title(f'Precision-Recall curve: {cfg.CLASS_NAME[class_id]}')
            plt.plot(pre_rec[:, 1], pre_rec[:, 0], color='g')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.axis([0, 1.05, 0, 1.05])
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(cfg.OUTPUT_PATH, f'precision_recall_curve_{cfg.CLASS_NAME[class_id]}.png'))
            plt.show()

        # Tn points.
        threshold = np.unique(pre_rec[:, 1])

        # Sum of below region.
        val_AP = 0

        for tdx, th in enumerate(threshold):
            # Calculate portion(A) divided by threshold.
            if tdx == 0:
                val_AP += np.max(pre_rec[:, 0][pre_rec[:, 1] <= th]) * th

            else:
                # Calculate portion(B, ...) divided by threshold.
                val_AP += np.max(pre_rec[:, 0][(threshold[tdx - 1] < pre_rec[:, 1]) & (pre_rec[:, 1] <= th)]) * (th - threshold[tdx - 1])

        return val_AP
