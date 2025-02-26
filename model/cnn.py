import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2)  # 2x2 pooling halves spatial size

        self.dropout = nn.Dropout(0.5)

        # Compute final feature map size (assuming input size is [batch, 1, 128, 1])
        final_height = 128 // 8  # Three max pools reduce height (128 → 64 → 32 → 16)
        final_width = 1  # Width remains 1

        self.fc1 = nn.Linear(64 * final_height * final_width, 128)  
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)  

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)  

        x = self.conv3(x)  # Added conv3
        x = self.bn3(x)  # Added bn3
        x = nn.ReLU()(x)
        x = self.pool(x)  

        x = x.view(x.size(0), -1)  # Flatten
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x



