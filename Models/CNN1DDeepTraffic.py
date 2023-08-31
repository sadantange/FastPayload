import torch.nn as nn
import torch
from torchinfo import summary
import torch.nn.functional as F

class IDCNN(nn.Module):
    def __init__(self, class_num):
        super(IDCNN, self).__init__()
        # two convolution, then one max pool
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=25,
                stride=1,
                padding = 'same'
            ),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool1d(
                kernel_size=3,
                stride=3,
                padding=1
            ),
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=25,
                stride=1,
                padding = 'same'
            ),
            nn.ReLU(),
        )
        self.output_net = nn.Sequential(
            nn.MaxPool1d(
                kernel_size=3,
                stride=3,
                padding=1
            ),
            nn.Flatten(),
            nn.Linear(in_features=88*64, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=class_num)
        )
    
    def forward(self, x):
        o = self.conv1(x)
        o = self.conv2(o)
        o = self.output_net(o)
        return o

if __name__ == "__main__":
    IDCNN = IDCNN(class_num=9)
    input_shape = (1, 1, 784)
    summary(IDCNN, input_shape)
