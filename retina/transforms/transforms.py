import numpy as np
import retina.transforms.functional as F


class Preprocess(object):
    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes = targets[:, :4].copy()
        labels = targets[:, -1].copy()
        landm = targets[:, 4:-1].copy()

        image_t, boxes_t, labels_t, landm_t, pad_image_flag = F.crop(image, boxes, labels, landm, self.img_dim)
        image_t = F.distort(image_t)
        image_t = F.pad_to_square(image_t,self.rgb_means, pad_image_flag)
        image_t, boxes_t, landm_t = F.mirror(image_t, boxes_t, landm_t)
        height, width, _ = image_t.shape
        image_t = F.resize_subtract_mean(image_t, self.img_dim, self.rgb_means)
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height

        landm_t[:, 0::2] /= width
        landm_t[:, 1::2] /= height

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, landm_t, labels_t))

        return image_t, targets_t
