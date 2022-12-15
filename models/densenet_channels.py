import torch.nn as nn
from .classification import TorchVisionModel
from torchvision import models

class DenseNet121_Variant(TorchVisionModel):
    def __init__(self, model_args, num_input_channels=3):
        super().__init__(models.densenet121, model_args)
        self.model.features.conv0 = nn.Conv2d(num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

class DenseNet121_naip(DenseNet121_Variant):
    def __init__(self, model_args=None):
        super().__init__(model_args, num_input_channels=4)

class DenseNet121_sentinel2(DenseNet121_Variant):
    def __init__(self, model_args=None):
        super().__init__(model_args, num_input_channels=4)

class DenseNet121_sentinel1(DenseNet121_Variant):
    def __init__(self, model_args=None):
        super().__init__(model_args, num_input_channels=2)

class DenseNet121_all(DenseNet121_Variant):
    def __init__(self, model_args=None):
        super().__init__(model_args, num_input_channels=10)

class DenseNet121_sentinels(DenseNet121_Variant):
    def __init__(self, model_args=None):
        super().__init__(model_args, num_input_channels=6)
