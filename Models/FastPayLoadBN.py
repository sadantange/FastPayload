import torch.nn as nn
import torch
from torchinfo import summary
import torch.nn.functional as F
from ModelBlock import GCBlock
    
class Block(nn.Module):
    def __init__(self, dim, ks, factor):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=ks, padding='same', groups=dim) # depthwise conv
        self.norm = nn.BatchNorm1d(dim)
        self.pwconv1 = nn.Linear(dim, factor * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.ReLU()
        self.pwconv2 = nn.Linear(factor * dim, dim)
        self.gc = GCBlock(dim)
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1) # (N, C, L) -> (N, L, c)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1) # (N, L, C) -> (N, C, L)
        x = self.gc(x)
        x = input + x
        return x

class IDCNN(nn.Module):
    def __init__(self, class_num, depths=[2, 3, 4, 2], dims=[128, 128, 64, 32]):
        super(IDCNN, self).__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=dims[0],
                kernel_size=4,
                stride=4
            ),
            nn.BatchNorm1d(dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm1d(dims[i]),
                nn.Conv1d(dims[i], dims[i+1], kernel_size=4, stride=4),
            )
            self.downsample_layers.append(downsample_layer)
        self.stages = self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dims[i], 8, 2) for j in range(depths[i])]
            )
            self.stages.append(stage)
        # finally, output layer
        self.head = nn.Linear(160, class_num)
    
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x.view(x.size(0), -1)
            
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    IDCNN = IDCNN(class_num=9)
    input_shape = (1, 1, 1500)
    summary(IDCNN, input_shape)
