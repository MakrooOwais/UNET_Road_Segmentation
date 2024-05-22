import torch
import pytorch_lightning as pl

from model import UNET
from data_module import DataModule


LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 125
NUM_WORKERS = 2
EARLY_STOPPING_PATIENCE = 5
PIN_MEMORY = True
LOAD_MODEL = False
DIR = "D:\\Machine Learning\\U-Net Segmentation\\Dataset"


def main():
    model = UNET(3, 1, lr=LEARNING_RATE)
    model.load_state_dict(
        torch.load("D:\\Machine Learning\\U-Net Segmentation\\saved_models\\model.pth")
    )
    data_module = DataModule(DIR, NUM_WORKERS, BATCH_SIZE, PIN_MEMORY)
    data_module.setup("validate")

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=NUM_EPOCHS,
        devices=[0],
        log_every_n_steps=1,
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    torch.save(
        model.state_dict(),
        "D:\\Machine Learning\\U-Net Segmentation\\saved_models\\model.pth",
    )


if __name__ == "__main__":
    main()
