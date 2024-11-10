import torch
from torch import nn

class DuelNet(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()

        self.in_channels = in_channels
        self.num_actions = num_actions

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1
            ), # 64
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1), # 32

            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1
            ), # 32
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1) # 16
        )

        self.flatten = nn.Flatten()

        self.linear_value = nn.Sequential(
            nn.Linear(16 * 16 * 16, 128),
            nn.ReLU()
        )
        self.value = nn.Linear(128, 1)

        self.linear_advantage = nn.Sequential(
            nn.Linear(16 * 16 * 16, 128),
            nn.ReLU()
        )
        self.advantage = nn.Linear(128, num_actions)
    
    def forward(self, x):
        y_conv = self.conv(x)
        y_flatten = self.flatten(y_conv)

        value_ = self.linear_value(y_flatten)
        value_ = self.value(value_)

        advantages_ = self.linear_advantage(y_flatten)
        advantages_ = self.advantage(advantages_)

        y = advantages_ - advantages_.mean(dim=1, keepdim=True)
        y = value_ + y

        return y

if __name__ == "__main__":
    x = torch.randn(4, 64, 64)
    x = x.unsqueeze(0)

    model = DuelNet(4, 10).eval()

    y = model(x)
    print(y.shape, y)

