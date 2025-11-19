# model_mobilenetv2.py
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def MobileNetV2_Custom(num_classes=2, pretrained=False):
    """
    MobileNetV2 з заміною останнього шару під 2 класи.
    """
    if pretrained:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        model = mobilenet_v2(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
