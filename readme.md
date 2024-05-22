## UNet: Road Segmentation with PyTorch Lightning

This code implements a UNet architecture for semantic segmentation, specifically trained on a road segmentation dataset using PyTorch Lightning.

**Encoder-Decoder Framework**

The UNet architecture follows the encoder-decoder paradigm.

* **Encoder:** The encoder part captures the contextual information from the input image. It consists of repeated applications of convolutional layers, often with max pooling operations to reduce the spatial dimensions while increasing the number of feature channels. This process effectively compresses the input image into a latent representation containing the essential features for road segmentation.
* **Decoder:** The decoder part aims to reconstruct the segmentation mask from the encoded representation. It performs upsampling operations to increase the spatial resolution and combines these upsampled features with the skipped connections from the encoder at corresponding scales. These skipped connections allow the decoder to recover precise spatial information lost during downsampling in the encoder. Finally, convolutional layers are used to refine the segmentation mask. 

**Implementation Details**

This code defines a `UNET` class that inherits from `pl.LightningModule`. Here's a breakdown of the key components:

* `DoubleConv`: This is a custom module that performs two consecutive convolutional layers, each followed by batch normalization and ReLU activation.
* `UNET`:
    * The `__init__` method defines the network architecture with `features` specifying the number of channels at each level.
    * `forward` method passes the input image through the encoder, bottleneck, and decoder to generate the segmentation mask. It utilizes skip connections to preserve spatial information.
    * `training_step`, `validation_step`, and `test_step` define the training, validation, and testing routines using PyTorch Lightning hooks.
    * `on_test_epoch_end` calculates the Dice score and accuracy metrics for the test set.
    * `configure_optimizers` configures the Adam optimizer with the learning rate specified in the `__init__` method.

**Sample Outputs**
<div align="center">
	<img src="https://github.com/MakrooOwais/UNET_Road_Segmentation/blob/main/Outputs/0.jpg">
   <img src="https://github.com/MakrooOwais/UNET_Road_Segmentation/blob/main/Outputs/1.jpg">
   <img src="https://github.com/MakrooOwais/UNET_Road_Segmentation/blob/main/Outputs/2.jpg">
</div>

**Running the Model**

This code requires PyTorch Lightning to be installed (`pip install pytorch-lightning`). You can train the model by creating a `DataModule` class for handling your road segmentation dataset and training loop, and then training the `UNET` model using a PyTorch Lightning trainer.

**References**

* Paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* Dataset: [road_segmentationv1.1_dataset](https://universe.roboflow.com/road-segmentation/road_segmentationv1.1)

This is a well-structured implementation of UNet with PyTorch Lightning. Feel free to adapt it further for your specific dataset and experiment with different hyperparameters!

## UNet: Road Segmentation with PyTorch Lightning

This code implements a UNet architecture for semantic segmentation, specifically trained on a road segmentation dataset using PyTorch Lightning.
