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

from pathlib import Path
from collections import OrderedDict
from . import functional as MF

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, in_channels=None, out_channels=None, 
                 backbone_name="mobilenet0.25", backbone_pretrained=True, 
                 backbone_weight="../weights/mobilenetV1X0.25_pretrain.tar",
                 phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()
        self.phase = phase
        if cfg != None:
            self.backbone_name = cfg["name"]
            self.backbone_pretrained = cfg["pretrain"]
            self.backbone_weight = backbone_weight
            
            self.in_channels = cfg['in_channel']
            self.out_channels = cfg['out_channel']
            
        else:
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.backbone_name = backbone_name
            self.backbone_pretrained = backbone_pretrained
            self.backbone_weight = backbone_weight
            
        in_channels_list = [self.in_channels * 2, self.in_channels * 4, self.in_channels * 8]
        
        self.body = self._make_body()
        self.fpn = FeaturePyramidNetwork(in_channels_list, self.out_channels)
        self.ssh1 = SingleStageHeadless(self.out_channels, self.out_channels)
        self.ssh2 = SingleStageHeadless(self.out_channels, self.out_channels)
        self.ssh3 = SingleStageHeadless(self.out_channels, self.out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, in_channels=self.out_channels)
        self.BboxHead = self._make_bbox_head(fpn_num=3, in_channels=self.out_channels)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, in_channels=self.out_channels)
        
    def _load_resnet50(self, pretrained=False):
        import torchvision.models as models
        backbone = models.resnet50(pretrained=pretrained)
        return backbone
    
    def _load_mobilenet_v1(self, pretrained=False):
        backbone = MobileNetV1()
        if pretrained and self.backbone_weight!=None:
            weight_path = Path(self.backbone_weight)
            checkpoint = torch.load(str(weight_path), map_location=torch.device('cpu'))
            
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:]  # remove module.
                new_state_dict[name] = v
            # load params
            backbone.load_state_dict(new_state_dict)
        
        return backbone
    
    def _make_body(self):
        backbone = None
        if self.backbone_name == 'mobilenet':
            backbone = self._load_mobilenet_v1(pretrained=self.backbone_pretrained)
            usage_layer = {'stage1': 1, 'stage2': 2, 'stage3': 3}
        elif self.backbone_name == 'resnet':
            backbone = self._load_resnet50(pretrained=self.backbone_pretrained)
            usage_layer = {'layer2': 1, 'layer3': 2, 'layer4': 3}
        else:
            raise ValueError("only mobilenet and resnet backbone are supported!")
            
        body = _utils.IntermediateLayerGetter(backbone, usage_layer)
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
    
    
def retina_face(pretrained=False, backbone="mobilenet", backbone_weight=None, final_weight=None):
    if backbone == 'mobilenet':
        model = RetinaFace(in_channels=32, out_channels=64, 
                           backbone_name="mobilenet", backbone_pretrained=pretrained,
                           backbone_weight=backbone_weight)
        if final_weight is not None:
            model = MF.load_model(model, pretrained_path=final_weight, load_to_cpu=True)
        
    elif backbone == "resnet":
        model = RetinaFace(in_channels=256, out_channels=256, 
                           backbone_name="resnet", backbone_pretrained=pretrained,
                           backbone_weight=backbone_weight)
        if final_weight is not None:
            model = MF.load_model(model, pretrained_path=final_weight, load_to_cpu=True)
    else:
        raise ValueError("only mobilenet and resnet backbone are supported!")
    
    return model
        
    