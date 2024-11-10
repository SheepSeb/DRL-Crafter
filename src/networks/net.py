import torch
from torch import nn

class ConvModel(nn.Module):
    def __init__(
        self,
        in_features,
        num_actions
    ):
        super().__init__()
        self.in_features = in_features
        self.num_actions = num_actions

        conv = nn.Sequential(
            nn.Conv2d(
                in_features,
                16,
                (3, 3),
                padding=1,
            ), # 64
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

        self._net = nn.Sequential(
            conv,
            classifier
        )

    def forward(self, x):
        return self._net(x)

if __name__ == "__main__":
    x = torch.randn(4, 64, 64)
    x = x.unsqueeze(0)

    model = ConvModel(in_features=4, num_actions=10).eval()
    y = model(x)

    print(y.shape, y)
