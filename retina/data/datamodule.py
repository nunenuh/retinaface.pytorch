import pytorch_lightning as pl
from typing import *

from .widerface import WiderFaceCustomDataset, detection_collate
from retina.transforms import transforms as CT
from pathlib import Path
from torch.utils.data import DataLoader



class WiderFaceDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/", image_size=640, means=(104, 117, 123),
                 batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.means = means
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def _pair_transform(self):
        tmft = CT.PairCompose([
            CT.CropDistort(image_size=self.image_size, means=self.means),
            CT.Mirror(),
            CT.SubtractMeanResize(image_size=self.image_size, means=self.means),
            CT.ToTensor()
            # CT.Preprocess(img_dim=self.image_size, rgb_means=self.means)
        ])
        
        return tmft
    
    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            data_path = Path(self.data_dir).joinpath("train")
            
            self.trainset = WiderFaceCustomDataset(root=str(data_path), train=True,  pair_transform=self._pair_transform())
            self.validset = WiderFaceCustomDataset(root=str(data_path), train=False, pair_transform=self._pair_transform())
        
        if stage in (None, "test"):
            test_path = Path(self.data_dir).joinpath("test")
            self.testset = WiderFaceCustomDataset(root=str(test_path), pair_transform=self._pair_transform())
        
        
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True,
                          collate_fn=detection_collate)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False,
                          collate_fn=detection_collate)

    def test_dataloader(self):
        return DataLoader(self.testset,batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False,
                          collate_fn=detection_collate)

