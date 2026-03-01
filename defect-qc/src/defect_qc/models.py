import torch
import torch.nn as nn
import torch.nn.functional as F



class QCCNNBase(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=3, stride=2, padding=1) 
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # [B, 4, 112, 112]
        x = F.relu(self.conv2(x)) # [B, 16, 56, 56]
        x = F.relu(self.conv3(x)) # [B, 64, 28, 28]
        x = F.relu(self.conv4(x)) # [B, 128, 14, 14]
        x = F.relu(self.conv5(x)) # [B, 256, 7, 7]
        x = self.pool(x)          # [B, 256, 1, 1]
        x = torch.flatten(x, 1)   # [B, 256]
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # logits
        return x
