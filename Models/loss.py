import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
import torch
import sys
sys.path.append(r"D:\\tmp\\IDS\\Constant")
from Constant import const
import numpy as np

# CBTerm1
def CBTerm(class_num_list, beta=0.999, use_cuda=True):
    if use_cuda:
        class_num_list = torch.tensor(class_num_list, dtype=torch.float).to(const.DEVICE)
    else:
        class_num_list = torch.tensor(class_num_list, dtype=torch.float)
    CBTerm = (1 - beta) / (1 - torch.pow(beta, class_num_list))
    return CBTerm

# CBTerm3
def cal_class_weights_one(per_class_num):
    maxnum = max(per_class_num)
    return torch.FloatTensor([maxnum / ele for ele in per_class_num]).cuda()

# CBTerm2
def cal_class_weights_two(per_class_num):
    return torch.FloatTensor([1 - 1 / ele for ele in per_class_num]).cuda()

class FocalLoss(nn.Module):
    def __init__(self,
                 weight: Optional[Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=weight, reduction='none')

    def __repr__(self):
        arg_keys = ['weight', 'gamma', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        if len(y) == 0:
            return torch.tensor(0.)

        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class GHMC_Loss(nn.Module):
    def __init__(self, 
                 weight: Optional[Tensor] = None,
                 bins=10, 
                 alpha=0.5, 
                 reduction="mean",):
        super(GHMC_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None
        self.nll_loss = nn.NLLLoss(weight=weight, reduction='none')
        self.reduction = reduction
    
    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss_grad(self, x, target, log_p):
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, target]
        pt = log_pt.exp()
        return pt-1
    
    def forward(self, x, target):
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, target)

        g = torch.abs(self._custom_loss_grad(x, target, log_p)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = 1 / gd
        loss = beta[bin_idx].cuda() * ce
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
    
class Focal_Two_Stage_Loss(nn.Module):
    def __init__(self,
                 weight: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean'):
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.weighted_nll_loss = nn.NLLLoss(weight=weight, reduction='none')
        self.nll_loss = nn.NLLLoss(reduction='none')

    def __repr__(self):
        arg_keys = ['weight', 'gamma', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor, stage) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        if len(y) == 0:
            return torch.tensor(0.)

        log_p = F.log_softmax(x, dim=-1)
        if stage == 1:
            ce = self.nll_loss(log_p, y)
        else:
            ce = self.weighted_nll_loss(log_p, y)

        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class GHMC_Two_Stage_Loss(nn.Module):
    def __init__(self, 
                 weight,
                 bins=10, 
                 alpha=0.5, 
                 reduction="mean"):
        super(GHMC_Two_Stage_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None
        self.weighted_nll_loss = nn.NLLLoss(weight=weight, reduction='none')
        self.nll_loss = nn.NLLLoss(reduction='none')
        self.reduction = reduction
    
    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss_grad(self, x, target, log_p):
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, target]
        pt = log_pt.exp()
        return pt-1
    
    def forward(self, x, target, stage):
        log_p = F.log_softmax(x, dim=-1)
        if stage == 1:
            ce = self.nll_loss(log_p, target)
        else:
            ce = self.weighted_nll_loss(log_p, target)

        g = torch.abs(self._custom_loss_grad(x, target, log_p)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = 1 / gd
        loss = beta[bin_idx].cuda() * ce
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
    

class CE_Two_Stage_Loss(nn.Module):
    def __init__(self,
                 weight,
                 reduction="mean") -> None:
        super(CE_Two_Stage_Loss, self).__init__()
        self.weighted_ce = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.ce = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(self, x, target, stage):
        if stage == 1:
            loss = self.ce(x, target)
        else:
            loss = self.weighted_ce(x, target)
        return loss

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=1):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list)) # nj的四次开方
        m_list = m_list * (max_m / np.max(m_list)) # 常系数 C
        m_list = torch.cuda.FloatTensor(m_list) # 转成 tensor
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8) # 和 x 维度一致全 0 的tensor
        index.scatter_(1, target.data.view(-1, 1), 1) # dim idx input
        index_float = index.type(torch.cuda.FloatTensor) # 转 tensor
        ''' 以上的idx指示的应该是一个batch的y_true '''
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m # y 的 logit 减去 margin
        output = torch.where(index, x_m, x) # 按照修改位置合并
        return F.cross_entropy(self.s*output, target)


    