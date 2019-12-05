import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_targets=10, num_channels=1):
        super(SimpleCNN, self).__init__()

        self.num_targets = num_targets
        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, num_targets, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_targets),
        )
        self.average_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.main(x)
        x = self.average_pooling(x)
        x = x.reshape(-1, self.num_targets)
        return x
