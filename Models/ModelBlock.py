import torch.nn as nn
import torch

class CBAMChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CBAMChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class CBAMSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(CBAMSpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class CBAMBlock(nn.Module):
    def __init__(self, dim):
        super(CBAMBlock, self).__init__()
        self.ca = CBAMChannelAttention(dim)
        self.sa = CBAMSpatialAttention()
    
    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _= x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class GCBlock(nn.Module):
    def __init__(self, channel, ratio=8):
        super(GCBlock, self).__init__()
        self.conx = nn.Conv1d(channel, 1, kernel_size=1, bias=False)
        self.act = nn.Softmax()
        self.tran = nn.Sequential(
                        nn.Conv1d(channel, channel // ratio, 1, bias=False),
                        nn.LayerNorm([channel // ratio, 1]),
                        nn.ReLU(),
                        nn.Conv1d(channel // ratio, channel, 1, bias=False)
                    )
    
    def contex_modeling(self, x):
        b, _, l = x.size()
        y = self.conx(x).view(b, l, 1)
        y = self.act(y)
        x = torch.matmul(x, y)
        return x

    def forward(self, x):
        z = self.contex_modeling(x)
        x = x + self.tran(z)
        return x

class ECABlock(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _= x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y).view(b, c, 1)
        return x * y.expand_as(x)
