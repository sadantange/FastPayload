import torch.nn as nn
from torchinfo import summary

class IDCNN(nn.Module):
    def __init__(self, class_num):
        super(IDCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, bias=True, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, bias=True, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.output_net = nn.Sequential(
            nn.Linear(in_features=3136, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=class_num)
        )

    def forward(self, x):
        o = self.conv(x)
        o = self.output_net(o)
        return o

if __name__ == "__main__":
    IDCNN = IDCNN(class_num=9)
    input_shape = (1, 1, 28, 28)
    summary(IDCNN, input_shape)


