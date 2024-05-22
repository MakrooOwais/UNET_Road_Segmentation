import pytorch_lightning as pl
import os
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from dataset import UNET_Dataset

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dir: str,
        num_workers: int,
        batch_size: int = 256,
        pin_memory=True,
    ):
        super().__init__()
        self.dir = dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

        self.val_transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )
        self.pin_memory = pin_memory

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train = UNET_Dataset(
                dir=os.path.join(self.dir, "train"),
                transform=self.train_transform,
            )

        if stage == "validate":
            self.val = UNET_Dataset(
                dir=os.path.join(self.dir, "valid"),
                transform=self.val_transform,
            )

        if stage == "test":
            self.test = UNET_Dataset(
                dir=os.path.join(self.dir, "test"), transform=self.val_transform
            )

        if stage == "predict":
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        pass


if __name__ == "__main__":
    data_module = DataModule("Dataset", 5, 64)
    data_module.setup('fit')
    print(len(data_module.train))
    data_module.setup('validate')
    print(len(data_module.val))