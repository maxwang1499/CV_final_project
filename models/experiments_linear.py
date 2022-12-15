import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from util import constants as C
    
class DenseNet121_multiTaskNet_split(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Number of hidden parameters after DenseNet121
        hidden_size = 256
        
        densenet121 = models.densenet121(pretrained=pretrained)   

        # Use DenseNet except last layer
        self.features = nn.Sequential(*list(densenet121.children())[:-1])
        
        # Number of input features in classifier layer
        in_features =  densenet121.classifier.in_features
        
        # After that, each task gets its own hidden layer + classifier layer
        self.fc_CAFOs = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, 1)
        )
        
        self.fc_Mines = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, 1)
        )
            
        self.fc_WWTreatment = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, 1)
        )
        
        self.fc_RefineriesAndTerminals = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, 1)
        )
        
        self.fc_Landfills = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, 1)
        )

        self.fc_ProcessingPlants = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, 1)
        )

        # Xavier initialization for all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain = 1)

    def forward(self, input_imgs):
        # Pass into DenseNet backbone
        output = self.features(input_imgs)
        output = F.relu(output, inplace=True)
        output = F.adaptive_avg_pool2d(output, (1, 1))
        output = torch.flatten(output, 1)
        
        # Pass into task-specific layers
        prd_CAFOs = self.fc_CAFOs(output)
        prd_Mines = self.fc_Mines(output)
        prd_WWTreatment = self.fc_WWTreatment(output)
        prd_RefineriesAndTerminals = self.fc_RefineriesAndTerminals(output)
        prd_Landfills = self.fc_Landfills(output)
        prd_ProcessingPlants = self.fc_ProcessingPlants(output)
        
        # Concat according to class labels list
        logits_list = []
        for label in C.class_labels_list:
            logits_list.append(locals()["prd_" + label])            
        
        probs = torch.cat(logits_list, -1)
        
        return probs

class DenseNet121_multiTaskNet_shared(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Number of hidden parameters after DenseNet121
        hidden_size = 256
        
        # Pre-trained DenseNet121
        densenet121 = models.densenet121(pretrained=pretrained)   

        # Use DenseNet except last layer
        self.features = nn.Sequential(*list(densenet121.children())[:-1])
        
        # Number of input features in classifier layer
        in_features =  densenet121.classifier.in_features

        self.fc_Shared = nn.Sequential(
            nn.Linear(in_features, hidden_size * 6),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size * 6, 6)
        )

        # Xavier initialization for all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain = 1)

    def forward(self, input_imgs):
        # Pass into DenseNet backbone
        output = self.features(input_imgs)
        output = F.relu(output, inplace=True)
        output = F.adaptive_avg_pool2d(output, (1, 1))
        output = torch.flatten(output, 1)
        
        prd_shared = self.fc_Shared(output)

        return prd_shared


