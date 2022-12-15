import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from util import constants as C

class ResNet_TaskSpecificLayer(nn.Module):
    def __init__(self, task_name):
        super().__init__()
        self.task_name = task_name
        self.layer = nn.Sequential(
            nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(in_features=2048, out_features=1, bias=True)
    
    def forward(self, x):
        x = self.layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

class ResNet_ConvSplit(nn.Module):
    def __init__(self, model, hook_layer):
        super().__init__()
        
        self.resnet = model  
        
        # stuff with forward hooks
        named_modules = dict(self.resnet.named_modules())
        branch_layer = named_modules[hook_layer]

        def hook(module, in_val, out_val):
            self.branch_value = out_val

        branch_layer.register_forward_hook(hook)
        
        self.task_specific_layers = nn.ModuleList()
        for task in C.class_labels_list:
            tsl = ResNet_TaskSpecificLayer(task)
            self.task_specific_layers.append(tsl)
        
        # Xavier initialization for all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain = 1)    

    def forward(self, input_imgs):
        _ = self.resnet(input_imgs)
        
        outputs = []
        for layer in self.task_specific_layers:
            prd = layer(self.branch_value)
            outputs.append(prd)
        
        probs = torch.cat(outputs, -1)
        return probs

class ResNet50_ConvSplit(ResNet_ConvSplit):
    def __init__(self, pretrained=True):
        model = models.resnet50(pretrained=pretrained)
        hook_layer = 'layer4.2.bn2'
        super().__init__(model, hook_layer)

class ResNet101_ConvSplit(ResNet_ConvSplit):
    def __init__(self, pretrained=True):
        model = models.resnet101(pretrained=pretrained)
        hook_layer = 'layer4.2.bn2'
        super().__init__(model, hook_layer)
