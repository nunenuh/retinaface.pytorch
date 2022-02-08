import numpy as np
import torch
import retina.transforms.functional as CF
import torchvision.transforms.functional as F
import torchvision.transforms as T



class PairCompose(T.Compose):
    def __init__(self, transforms):
        super(PairCompose, self).__init__(transforms)

    def __call__(self, image, targets):
        for t in self.transforms:
            image, targets = t(image, targets)
        return image, targets
    
class CropDistort(object):
    def __init__(self, image_size=640, means=(104, 117, 123)):
        self.image_size = image_size
        self.means = means
    
    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"
        boxes, labels, landm = targets[:, :4].copy(), targets[:, -1].copy(), targets[:, 4:-1].copy()

        image_t, boxes_t, labels_t, landm_t, pad_image_flag = CF.crop(image, boxes, labels, landm, self.image_size)
        image_t = CF.distort(image_t)
        image_t = CF.pad_to_square(image_t,self.means, pad_image_flag)
        
        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, landm_t, labels_t))
        
        return image_t, targets_t
    
    
class Mirror(object):
    def __init__(self):
        ...

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes, labels, landm = targets[:, :4].copy(), targets[:, -1].copy(), targets[:, 4:-1].copy()

        image_t, boxes_t, landm_t = CF.mirror(image, boxes, landm)
        
        labels_t = np.expand_dims(labels, 1)
        targets_t = np.hstack((boxes_t, landm_t, labels_t))
        
        return image_t, targets_t
        
        
        
class SubtractMeanResize(object):
    def __init__(self, image_size=640, means=(104, 117, 123)):
        self.image_size = image_size
        self.means = means
        
    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes = targets[:, :4].copy()
        labels = targets[:, -1].copy()
        landm = targets[:, 4:-1].copy()
        
        height, width, _ = image.shape
        image_t = CF.resize_subtract_mean(image, self.image_size, self.means)
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height

        landm[:, 0::2] /= width
        landm[:, 1::2] /= height

        labels_t = np.expand_dims(labels, 1)
        targets_t = np.hstack((boxes, landm, labels_t))

        return image_t, targets_t

class ToTensor(object):
    def __init__(self):
        ...
        
    def __call__(self, image, targets):
        image = torch.from_numpy(image)
        # targets = torch.from_numpy(targets)
        return image, targets

class Preprocess(object):
    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes = targets[:, :4].copy()
        labels = targets[:, -1].copy()
        landm = targets[:, 4:-1].copy()

        image_t, boxes_t, labels_t, landm_t, pad_image_flag = CF.crop(image, boxes, labels, landm, self.img_dim)
        
        image_t = CF.distort(image_t)
        image_t = CF.pad_to_square(image_t,self.rgb_means, pad_image_flag)
        
        image_t, boxes_t, landm_t = CF.mirror(image_t, boxes_t, landm_t)
        
        height, width, _ = image_t.shape
        image_t = CF.resize_subtract_mean(image_t, self.img_dim, self.rgb_means)
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height

        landm_t[:, 0::2] /= width
        landm_t[:, 1::2] /= height

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, landm_t, labels_t))

        return image_t, targets_t



