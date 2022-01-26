import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from .backbone import MobileNetV1 as MobileNetV1
from .module import FeaturePyramidNetwork as FeaturePyramidNetwork
from .module import SingleStageHeadless as SingleStageHeadless
from .module import ClassHead, BboxHead, LandmarkHead


class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()
        self.phase = phase
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        
        self.body = self._make_body(cfg)
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels)
        self.ssh1 = SingleStageHeadless(out_channels, out_channels)
        self.ssh2 = SingleStageHeadless(out_channels, out_channels)
        self.ssh3 = SingleStageHeadless(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, in_channels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, in_channels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, in_channels=cfg['out_channel'])
        
    def _make_body(self, cfg):
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                import os
                from pathlib import Path
                base_path = os.getcwd()
                wpath = "../weights/mobilenetV1X0.25_pretrain.tar"
                weight_path = Path(base_path).joinpath(wpath)
                checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])
            
        body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        return body

    def _make_class_head(self,fpn_num=3, in_channels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(in_channels, anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3, in_channels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(in_channels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, in_channels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(in_channels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output