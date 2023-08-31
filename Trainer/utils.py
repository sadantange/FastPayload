import torchmetrics
from tqdm import tqdm
import torch

@torch.no_grad()
def cal_metrics(epoch, epochs, model, valid_dataloader, class_num):
    model.eval()
    All_acc = torchmetrics.Accuracy(task="multiclass", average=None, num_classes=class_num).cuda()
    Weightedacc = torchmetrics.Accuracy(task="multiclass", average='weighted', num_classes=class_num).cuda()
    Weightedf1 = torchmetrics.F1Score(task="multiclass", average='weighted', num_classes=class_num).cuda()
    Weightedprecision = torchmetrics.Precision(task="multiclass", average='weighted', num_classes=class_num).cuda()
    Macroacc = torchmetrics.Accuracy(task="multiclass", average='macro', num_classes=class_num).cuda()
    Macrof1 = torchmetrics.F1Score(task="multiclass", average='macro', num_classes=class_num).cuda()
    Macroprecision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=class_num).cuda()
    # 设置进度条
    bar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader))
    for i, (X, labels) in bar:
        X, labels = X.cuda(), labels.cuda()
        preds = model(X)
        All_acc(preds, labels)
        Weightedacc(preds, labels)
        Weightedf1(preds, labels)
        Weightedprecision(preds, labels)
        Macroacc(preds, labels)
        Macrof1(preds, labels)
        Macroprecision(preds, labels)
        bar.set_postfix(epoch=epoch, max_epoch=epochs, type="Validate")
    all_acc = All_acc.compute()
    all_acc = [i.item() for i in all_acc]
    weightedacc = Weightedacc.compute().item()
    weightedf1 = Weightedf1.compute().item()
    weightedprecision = Weightedprecision.compute().item()
    macroacc = Macroacc.compute().item()
    macrof1 = Macrof1.compute().item()
    macroprecision = Macroprecision.compute().item()
    return all_acc, weightedacc, weightedf1, weightedprecision, macroacc, macrof1, macroprecision

@torch.no_grad()
def cal_metrics_for_saving(epoch, epochs, model, valid_dataloader, class_num):
    model.eval()
    Macroacc = torchmetrics.Accuracy(task="multiclass", average='macro', num_classes=class_num).cuda()
    Weightedacc = torchmetrics.Accuracy(task="multiclass", average='weighted', num_classes=class_num).cuda()
    Macrof1 = torchmetrics.F1Score(task="multiclass", average='macro', num_classes=class_num).cuda()
    # 设置进度条
    bar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader))
    for i, (X, labels) in bar:
        X, labels = X.cuda(), labels.cuda()
        preds = model(X)
        Weightedacc(preds, labels)
        Macroacc(preds, labels)
        Macrof1(preds, labels)
        bar.set_postfix(epoch=epoch, max_epoch=epochs, type="Validate")
    weightedacc = Weightedacc.compute().item()
    macroacc = Macroacc.compute().item()
    macrof1 = Macrof1.compute().item()
    return weightedacc, macroacc, macrof1