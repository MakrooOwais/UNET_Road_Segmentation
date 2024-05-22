import torch
import os
import numpy as np

from pycocotools.coco import COCO
from torch.utils.data import Dataset
from PIL import Image


class UNET_Dataset(Dataset):
    def __init__(self, dir: str, transform=None):
        """
        Args:
            dir (string): Directory with the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dir = dir
        self.anno_file = [x for x in os.listdir(dir) if x.endswith("json")][0]
        self.transform = transform
        self.coco = COCO(os.path.join(dir, self.anno_file))

    def __len__(self):
        return len(self.coco.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_coco = self.coco.imgs[idx]
        # Read Image
        img_path = os.path.join(self.dir, img_coco["file_name"])
        image = np.array(Image.open(img_path).convert("RGB"))

        cat_ids = self.coco.getCatIds()
        anns_ids = self.coco.getAnnIds(
            imgIds=img_coco["id"], catIds=cat_ids, iscrowd=None
        )
        anns = self.coco.loadAnns(anns_ids)

        if anns:
            mask = self.coco.annToMask(anns[0])
        else:
            mask = np.zeros(image.shape[:-1])

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask.float()
