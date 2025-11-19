# model_cnnlite.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLite(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 224x224 → 112x112
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 112x112 → 56x56
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 56x56 → 28x28
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # 28x28 з 64 каналами → 64*28*28
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)      # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x