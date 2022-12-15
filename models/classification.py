import pretrainedmodels 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet 


class PretrainedModel(nn.Module):
    """Pretrained model, either from Cadene or TorchVision."""

    def __init__(self):
        super(PretrainedModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError('Subclass of PretrainedModel ' +
                                  'must implement forward method.')

    def fine_tuning_parameters(self, boundary_layers, lrs):
        """Get a list of parameter groups that can be passed to an optimizer.

        Args:
            boundary_layers: List of names for the boundary layers.
            lrs: List of learning rates for each parameter group, from earlier
            to later layers.

        Returns:
            param_groups: List of dictionaries, one per parameter group.
        """

        def gen_params(start_layer, end_layer):
            saw_start_layer = False
            for name, param in self.named_parameters():
                if end_layer is not None and name == end_layer:
                    # Saw the last layer -> done
                    return
                if start_layer is None or name == start_layer:
                    # Saw the first layer -> Start returning layers
                    saw_start_layer = True

                if saw_start_layer:
                    yield param

        if len(lrs) != boundary_layers + 1:
            raise ValueError(f'Got {boundary_layers + 1} param groups, ' +
                             f'but {lrs} learning rates')

        # Fine-tune the network's layers from encoder.2 onwards
        boundary_layers = [None] + boundary_layers + [None]
        param_groups = []
        for i in range(len(boundary_layers) - 1):
            start, end = boundary_layers[i:i + 2]
            param_groups.append({'params': gen_params(start, end),
                                 'lr': lrs[i]})
        return param_groups


class EfficientNetModel(PretrainedModel):
    """EfficientNet models:
    https://github.com/lukemelas/EfficientNet-PyTorch
    """

    def __init__(self, model_name, model_args=None):
        super().__init__()
        num_classes = model_args.get("num_classes", None)
        pretrained = model_args.get("pretrained", False)

        if pretrained:
            self.model = EfficientNet.from_pretrained(
                model_name, num_classes=num_classes)
        else:
            self.model = EfficientNet.from_name(
                model_name, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class CadeneModel(PretrainedModel):
    """Models from Cadene's GitHub page of pretrained networks:
        https://github.com/Cadene/pretrained-models.pytorch
    """

    def __init__(self, model_name, model_args=None):
        super(CadeneModel, self).__init__()

        model_class = pretrainedmodels.__dict__[model_name]
        pretrained = "imagenet" if model_args['pretrained'] else None
        self.model = model_class(num_classes=1000,
                                 pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)

        num_ftrs = self.model.last_linear.in_features
        self.fc = nn.Linear(num_ftrs, model_args['num_classes'])

    def forward(self, x):
        x = self.model.features(x)
        x = F.relu(x, inplace=False)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x


class TorchVisionModel(PretrainedModel):
    """Models from TorchVision's GitHub page of pretrained neural networks:
        https://github.com/pytorch/vision/tree/master/torchvision/models
    """

    def __init__(self, model_fn, model_args):
        super(TorchVisionModel, self).__init__()
        num_classes = model_args.get("num_classes", 1000)
        pretrained = model_args.get("pretrained", False)

        self.model = model_fn(num_classes=num_classes)
        if pretrained: 
            model_pretrained = model_fn(pretrained=True) 
            state_dict_pretrained = model_pretrained.state_dict()
            state_dict = self.model.state_dict()
            for k in state_dict_pretrained:
                if k in state_dict:
                    if state_dict[k].shape != state_dict_pretrained[k].shape:
                        print(f"Skip loading parameter: {k}, "
                        f"required shape: {state_dict[k].shape}, "
                        f"loaded shape: {state_dict_pretrained[k].shape}")
                    else:
                        state_dict[k] = state_dict_pretrained[k]
            self.model.load_state_dict(state_dict)
                
    def forward(self, x):
        return self.model(x)

class EfficientNetB7(EfficientNetModel):
    def __init__(self, model_args=None):
        super().__init__('efficientnet-b7', model_args)


class DenseNet121(TorchVisionModel):
    def __init__(self, model_args=None):
        super(DenseNet121, self).__init__(models.densenet121, model_args)


class ResNet101(TorchVisionModel):
    def __init__(self, model_args=None):
        super(ResNet101, self).__init__(models.resnet101, model_args)
    

class VGG16(TorchVisionModel):
    def __init__(self, model_args=None):
        super(VGG16, self).__init__(models.vgg16, model_args)


class Inceptionv3(CadeneModel):
    def __init__(self, model_args=None):
        super(Inceptionv3, self).__init__('inceptionv3', model_args)
