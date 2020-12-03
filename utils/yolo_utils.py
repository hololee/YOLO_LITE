from torchvision.transforms import functional as F


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


def calculate_loss(gpu, output, target, optimizer):
    '''
    :param gpu: device instance.
    :param output: yolo output feature.
    :param target: target information. {"boxes": boxes, "classes": classes}
                                        boxes_shape = [category, cell_x, cell_y, center_x center_y, w, h], classes_shape = [m]
     :param optimizer: yolo_optimizer
    :return:
    '''

    # Initialize loss.
    loss = 0

    # NOTICE: Prepare targets.
    # shape : [n_boxes, x, y, w, h]
    object_boxes = []

    # shape : [n_boxes, conf]
    object_conf = []
    non_object_conf = []

    # shape : [n_boxes, 0, 0, ... ,1 , .... , 0], object_prob[:, 1:] : one-hot vectors.
    object_prob = []

    # NOTICE: Prepare inputs.
    # shape : [n_boxes, x, y, w, h]
    object_boxes = []

    # shape : [n_boxes, conf]
    object_conf = []
    non_object_conf = []

    # shape : [n_boxes, 0, 0, ... ,1 , .... , 0], object_prob[:, 1:] : one-hot vectors.
    object_prob = []

    # For calculate below loss, find center of each G.T and find position in (7 x 7) grid.

    # NOTICE: Calculate Bounding Box loss.

    # NOTICE : Calculate object confidence loss.
    # should calculated for non-object also.

    # NOTICE: Calculate Class Probability loss.

    return loss
