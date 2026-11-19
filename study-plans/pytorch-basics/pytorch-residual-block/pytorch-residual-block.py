import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.act = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(channels, eps=0.00001)
        self.bn2 = torch.nn.BatchNorm2d(channels, eps=0.00001)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor with the same shape as x.
        """
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)

        y = self.conv2(y)
        y = self.bn2(y)
        
        y = y + x
        y = self.act(y)
        y = y.to(torch.float32)
        return y