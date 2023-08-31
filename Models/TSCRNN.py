import torch.nn as nn
from torchinfo import summary

class TSCRNN(nn.Module):
    def __init__(self, class_num):
        super(TSCRNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, kernel_size=3, out_channels=64, stride=1, padding='same'),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, kernel_size=3, out_channels=64, stride=1, padding='same'),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.bilstm1 = nn.LSTM(input_size=375, num_layers=1, hidden_size=256, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.bilstm2 = nn.LSTM(input_size=512, num_layers=1, hidden_size=256, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.output_net = nn.Sequential(
            nn.Linear(in_features=512, out_features=class_num)
        )

    def forward(self, x):
        o = self.conv1(x)
        o = self.conv2(o)
        o, (hn, cn) = self.bilstm1(o)
        o = self.dropout1(o)
        o, (hn, cn) = self.bilstm2(o)
        o = self.dropout2(o)
        o = self.output_net(o[:, -1])
        return o

if __name__ == "__main__":
    TSCRNN = TSCRNN(class_num=16)
    input_shape = (1, 1, 1500)
    summary(TSCRNN, input_shape)