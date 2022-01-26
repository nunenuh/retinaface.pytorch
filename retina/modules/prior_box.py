from re import I
import torch
from itertools import product as product
import numpy as np
from math import ceil


base_steps = [8, 16, 32]
base_min_sizes = [[16, 32], [64, 128], [256, 512]]

def prior_box(feature_maps, min_sizes, image_size, steps, clip):
    anchors = []
    for k, f in enumerate(feature_maps):
        mins_sizez = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in mins_sizez:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]

    # back to torch land
    output = torch.Tensor(anchors).view(-1, 4)
    if clip: output.clamp_(max=1, min=0)
    return output


class CustomPriorBox(object):
    def __init__(self, image_size, min_sizes=None, steps=None, clip=False):
        self.image_size = image_size
        if min_sizes == None:
            self.min_sizes = base_min_sizes
        else:
            self.min_sizes = min_sizes
            
        if steps==None:
            self.steps = base_steps
        else:
            self.steps = steps

        self.clip = clip
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"
    
    def forward(self):
        return prior_box(
            self.feature_maps, self.min_sizes, 
            self.image_size, self.steps, self.clip
        )

class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        output = prior_box(
            self.feature_maps, self.min_sizes, 
            self.image_size, self.steps, self.clip
        )
        return output
        
        # anchors = []
        # for k, f in enumerate(self.feature_maps):
        #     min_sizes = self.min_sizes[k]
        #     for i, j in product(range(f[0]), range(f[1])):
        #         for min_size in min_sizes:
        #             s_kx = min_size / self.image_size[1]
        #             s_ky = min_size / self.image_size[0]
        #             dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
        #             dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
        #             for cy, cx in product(dense_cy, dense_cx):
        #                 anchors += [cx, cy, s_kx, s_ky]

        # # back to torch land
        # output = torch.Tensor(anchors).view(-1, 4)
        # if self.clip:
        #     output.clamp_(max=1, min=0)
        # return output


