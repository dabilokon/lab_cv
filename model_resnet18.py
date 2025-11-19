# model_resnet18.py
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def ResNet18_Custom(num_classes: int = 2, pretrained: bool = True):
    
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = resnet18(weights=None)

    # заміна останнього повнозв'язного шару
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
