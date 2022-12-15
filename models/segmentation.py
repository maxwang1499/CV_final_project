import segmentation_models_pytorch as smp
from torch import nn


class SegmentationModel(nn.Module):
    """Segmentation model interface."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError('Subclass of PretrainedModel ' +
                                  'must implement forward method.')


class SMPModel(SegmentationModel):
    """
    PyTorch Segmentation models from
    https://github.com/qubvel/segmentation_models.pytorch
    """
    SMP_ARCHITECTURE_MAP = {
        "UNet": smp.Unet,
        "UNetPlusPlus": smp.UnetPlusPlus,
        "Linknet": smp.Linknet,
        "DeepLabV3": smp.DeepLabV3,
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "FPN": smp.FPN,
        "PSPNet": smp.PSPNet,
    }

    def __init__(
            self,
            model_args=None):
        num_classes = model_args.get("num_classes", None)
        num_channels= model_args.get("num_channels", 3)
        encoder_name = model_args.get("encoder", None) 
        encoder_weights = model_args.get("pretrained", None)
        super().__init__()
        architecture = self.__class__.__name__
        if architecture not in self.SMP_ARCHITECTURE_MAP.keys():
            raise ValueError(
                f"Unknown architecture of SMPModel. Please choose from {list(SMP_ARCHITECTURE_MAP.keys())}.")
        _model_fn = self.SMP_ARCHITECTURE_MAP[architecture]
        self.model = _model_fn(encoder_name=encoder_name,
                               encoder_weights=encoder_weights,
                               in_channels=num_channels,
                               classes=num_classes)

    def forward(self, x):
        return self.model(x)


class UNet(SMPModel):
    pass


class UNetPlusPlus(SMPModel):
    pass


class Linknet(SMPModel):
    pass


class DeepLabV3(SMPModel):
    pass


class DeepLabV3Plus(SMPModel):
    pass


class FPN(SMPModel):
    pass


class PSPNet(SMPModel):
    pass
