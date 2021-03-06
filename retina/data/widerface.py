import os
import os.path
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset 
from pathlib import Path
from sklearn.model_selection import train_test_split


class WiderFaceCustomDataset(Dataset):
    def __init__(self, root, transform=None, pair_transform=None, train=True, val_size=0.2,
                 label_filename: str = 'label.txt', images_dirname: str = 'images',):
        self.root: str = root
        self.label_filename: str = label_filename
        self.images_dirname: str = images_dirname
        self.train = train
        self.val_size = val_size
        
        self.root_path: Path = Path(root)
        self.label_path: Path = self.root_path.joinpath(self.label_filename)
        self.image_path: Path = self.root_path.joinpath(self.images_dirname)
        self.transform = transform
        self.pair_transform = pair_transform
        
        self.images, self.labels = self._build_data()
        self._do_split_validation()
        
       
    def _load_text_file(self, path):
        lines = []
        with open(str(path), 'r') as f:
            lines = f.readlines()
        return lines
    
    def _load_image(self, path):
        image = cv2.imread(str(path))
        return image
        
    def _build_data(self):
        lines = self._load_text_file(str(self.label_path))
        images, words, labels = [], [], []

        # print(lines[:10])
        is_first = True
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if is_first is True:
                    is_first = False
                else:
                    labels_copy = labels.copy()
                    words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = self.image_path.joinpath(path)
                images.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        words.append(labels)
        
        return images, words
    
    def _do_split_validation(self):
        train_images, valid_images, train_labels, valid_labels = train_test_split(
            self.images, self.labels,
            test_size=self.val_size, 
            random_state=1261
        )
        
        self.train_images = train_images
        self.train_labels = train_labels
        self.valid_images = valid_images
        self.valid_labels = valid_labels
        
        if self.train:
            self.images = self.train_images
            self.labels = self.train_labels
        else:
            self.images = self.valid_images
            self.labels = self.valid_labels
    
    
    def _label_to_annotation(self, labels):
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        return target

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        labels = self.labels[index]
        impath = self.images[index]
        
        image = self._load_image(impath)
        target = self._label_to_annotation(labels)
        
        if self.transform:
            image = self.transform(image)
        
        if self.pair_transform:
            image, target = self.pair_transform(image, target)
        

        return image, target


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
