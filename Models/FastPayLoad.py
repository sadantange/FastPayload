import torch.nn as nn
import torch
from torchinfo import summary
import torch.nn.functional as F
from ModelBlock import GCBlock
from ModelBlock import CBAMBlock
from ModelBlock import SEBlock
from ModelBlock import ECABlock

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x
    
class Block(nn.Module):
    def __init__(self, dim, ks, factor):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=ks, padding='same', groups=dim) # depthwise conv
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, factor * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.ReLU()
        self.pwconv2 = nn.Linear(factor * dim, dim)
        self.gc = GCBlock(dim)
        # self.gc =CBAMBlock(dim)
        # self.gc = SEBlock(dim)
        # self.gc = ECABlock(dim)
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1) # (N, C, L) -> (N, L, c)
        x = self.norm(x)
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
            LayerNorm(dims[0], data_format="channels_first")
            # nn.LayerNorm([dims[0], 375])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-5, data_format="channels_first"),
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
