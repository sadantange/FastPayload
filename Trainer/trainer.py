from tqdm import tqdm
import gc
import torch
import os
from Constant import const
from Trainer.utils import cal_metrics
from Trainer.utils import cal_metrics_for_saving
import glob
import numpy as np

class Trainer(object):
    def __init__(self, model, criterion, optimizer, scheduler, train_dataloader, valid_dataloader, class_num, epochs, save_path=const.SAVE_PATH):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.class_num = class_num
        self.epochs = epochs
        self.save_path=save_path
        self.model = self.model.cuda()

    def train_one_epoch(self, epoch, optimizer, criterion):

        # 把模型设置为训练模式
        self.model.train()

        # 设置进度条
        bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
        # 计算一个epoch中的数据数量
        dataset_size = 0.0

        # 记录计算一个epoch的累计损失
        running_loss = 0.0

        correct = 0

        for step, (inputs, labels) in bar:
            # batchsize等于一个batch中inputs的数量
            batch_size = inputs.size(0)
            
            inputs, labels = inputs.cuda(), labels.cuda()

            # forward + backward + optimze
            outputs = self.model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            # zero the parameter gradients
            # 根据pytorch中backward（）函数的计算，当网络参量进行反馈时，
            # 梯度是累积计算而不是被替换，但在处理每一个batch时并不需要与其他batch的梯度混合起来累积计算，
            # 因此需要对每个batch调用一遍zero_grad（）将参数梯度置0
            optimizer.zero_grad()

            # 将每一个样本的损失加到总的一个epoch的损失中去
            running_loss += (loss.item() * batch_size)
            # 将batchsize加到总的一个epoch的数据数量中
            dataset_size += batch_size
            # running_loss的平均
            epoch_loss = running_loss / dataset_size
            # training accuracy
            pred = torch.argmax(outputs, dim=1)
            correct += torch.eq(pred, labels).sum().float().item()
            acc = correct / dataset_size

            # 设置进度条的额外情况
            bar.set_postfix(acc=acc, type="Train", epoch=epoch, max_epoch=self.epochs, LR=optimizer.param_groups[0]['lr'], train_loss=epoch_loss)
        # 进行垃圾回收
        gc.collect()
        return epoch_loss

    def run_training(self):
        # 训练开始
        for epoch in range(1, self.epochs + 1):
            # 计算一个epoch loss
            loss = self.train_one_epoch(epoch, self.optimizer, self.criterion)
            all_acc, weightedacc, weightedf1, weightedprecision, macroacc, macrof1, macroprecision = cal_metrics(epoch, self.epochs, self.model, self.valid_dataloader, self.class_num)
            print('Weighted Acc: ', weightedacc, 'Weighted F1: ', weightedf1, 'Weighted Precision: ', weightedprecision)
            print('Macro Acc: ', macroacc, 'Macro F1: ', macrof1, 'Macro Precision: ', macroprecision)
            print('All class accuracy: ', all_acc)
            self.scheduler.step()
    
    def run_training_and_valid_once(self):
        # 训练开始
        for epoch in range(1, self.epochs + 1):
            # 计算一个epoch loss
            loss = self.train_one_epoch(epoch, self.optimizer, self.criterion)
            self.scheduler.step()
        all_acc, weightedacc, weightedf1, weightedprecision, macroacc, macrof1, macroprecision = cal_metrics(epoch, self.epochs, self.model, self.valid_dataloader, self.class_num)
        print('Weighted Acc: ', weightedacc, 'Weighted F1: ', weightedf1, 'Weighted Precision: ', weightedprecision)
        print('Macro Acc: ', macroacc, 'Macro F1: ', macrof1, 'Macro Precision: ', macroprecision)
        print('All class accuracy: ', all_acc)
    
    # train and save
    def run_training_and_saving(self, model_name):
        # 训练开始
        best_metric = 0
        weightedacc_list = []
        macroacc_list = []
        model_path = os.path.join(self.save_path, model_name+'.pt')
        record_path = os.path.join(self.save_path, model_name+'_training_record.npy')
        for epoch in range(1, self.epochs + 1):
            # 计算一个epoch loss
            loss = self.train_one_epoch(epoch, self.optimizer, self.criterion)
            weightedacc, macroacc, macrof1 = cal_metrics_for_saving(epoch, self.epochs, self.model, self.valid_dataloader, self.class_num)
            if macrof1 > best_metric:
                best_metric = macrof1
                globlist = glob.glob(os.path.join(self.save_path, model_name+'*'))
                if(len(globlist) == 1):
                    os.remove(globlist[0])
                print('saving best model')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, model_path)
            weightedacc_list.append(weightedacc)
            macroacc_list.append(macroacc)
            self.scheduler.step()
        training_record = np.array([weightedacc_list, macroacc_list])
        np.save(record_path, training_record)

    # load and train and save
    def load_and_training_and_saving(self, checkpoint_path, model_name):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_metric = 0
        model_path = os.path.join(self.save_path, model_name+'.pt')
        for epoch in range(start_epoch+1, self.epochs + 1):
            # 计算一个epoch loss
            loss = self.train_one_epoch(epoch, self.optimizer, self.criterion)
            weightedacc, macroacc, macrof1 = cal_metrics_for_saving(epoch, self.epochs, self.model, self.valid_dataloader, self.class_num)
            if macrof1 > best_metric:
                best_metric = macrof1
                globlist = glob.glob(os.path.join(self.save_path, model_name+'*'))
                if(len(globlist) == 1):
                    os.remove(globlist[0])
                print('saving best model')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, model_path)
            self.scheduler.step()

class TwoStageTrainer(object):
    def __init__(self, model, criterion, optimizer, scheduler, train_dataloader, valid_dataloader, class_num, epochs, save_path=const.SAVE_PATH):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.class_num = class_num
        self.epochs = epochs
        self.save_path=save_path
        self.model = self.model.cuda()
    def train_one_epoch(self, epoch, stage):

        # 把模型设置为训练模式
        self.model.train()

        # 设置进度条
        bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
        # 计算一个epoch中的数据数量
        dataset_size = 0.0

        # 记录计算一个epoch的累计损失
        running_loss = 0.0

        correct = 0

        for step, (inputs, labels) in bar:
            # batchsize等于一个batch中inputs的数量
            batch_size = inputs.size(0)
            
            inputs, labels = inputs.cuda(), labels.cuda()

            # forward + backward + optimze
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.long(), stage)
            loss.backward()
            self.optimizer.step()

            # zero the parameter gradients
            # 根据pytorch中backward（）函数的计算，当网络参量进行反馈时，
            # 梯度是累积计算而不是被替换，但在处理每一个batch时并不需要与其他batch的梯度混合起来累积计算，
            # 因此需要对每个batch调用一遍zero_grad（）将参数梯度置0
            self.optimizer.zero_grad()

            # 将每一个样本的损失加到总的一个epoch的损失中去
            running_loss += (loss.item() * batch_size)
            # 将batchsize加到总的一个epoch的数据数量中
            dataset_size += batch_size
            # running_loss的平均
            epoch_loss = running_loss / dataset_size
            # training accuracy
            pred = torch.argmax(outputs, dim=1)
            correct += torch.eq(pred, labels).sum().float().item()
            acc = correct / dataset_size

            # 设置进度条的额外情况
            bar.set_postfix(stage=stage, acc=acc, type="Train", epoch=epoch, max_epoch=self.epochs, LR=self.optimizer.param_groups[0]['lr'], train_loss=epoch_loss)
        # 进行垃圾回收
        gc.collect()
        return epoch_loss

    def run_training(self):
        # 训练开始
        for epoch in range(1, self.epochs + 1):
            if (epoch % 2) == 0:
                # stage 2
                self.optimizer.param_groups[0]['lr']=0.1*self.optimizer.param_groups[0]['lr']
                loss = self.train_one_epoch(epoch, 2)
                self.optimizer.param_groups[0]['lr']=10*self.optimizer.param_groups[0]['lr']
                all_acc, weightedacc, weightedf1, weightedprecision, macroacc, macrof1, macroprecision = cal_metrics(epoch, self.epochs, self.model, self.valid_dataloader, self.class_num)
                print('Weighted Acc: ', weightedacc, 'Weighted F1: ', weightedf1, 'Weighted Precision: ', weightedprecision)
                print('Macro Acc: ', macroacc, 'Macro F1: ', macrof1, 'Macro Precision: ', macroprecision)
                print('All class accuracy: ', all_acc)
            else:
                # stage 1
                loss = self.train_one_epoch(epoch, 1)
            self.scheduler.step()
    
    # train and save
    def run_training_and_saving(self, model_name):
        # 训练开始
        best_metric = 0
        weightedacc_list = []
        macroacc_list = []
        model_path = os.path.join(self.save_path, model_name+'.pt')
        record_path = os.path.join(self.save_path, model_name+'_training_record.npy')
        for epoch in range(1, self.epochs + 1):
            if (epoch % 2) == 0:
                # stage 2
                self.optimizer.param_groups[0]['lr']=0.1*self.optimizer.param_groups[0]['lr']
                loss = self.train_one_epoch(epoch, 2)
                self.optimizer.param_groups[0]['lr']=10*self.optimizer.param_groups[0]['lr']
            else:
                # stage 1
                loss = self.train_one_epoch(epoch, 1)
            weightedacc, macroacc, macrof1 = cal_metrics_for_saving(epoch, self.epochs, self.model, self.valid_dataloader, self.class_num)
            if macrof1 > best_metric:
                best_metric = macrof1
                globlist = glob.glob(os.path.join(self.save_path, model_name+'*'))
                if(len(globlist) == 1):
                    os.remove(globlist[0])
                print('saving best model')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                }, model_path)
            weightedacc_list.append(weightedacc)
            macroacc_list.append(macroacc)
            self.scheduler.step()
        training_record = np.array([weightedacc_list, macroacc_list])
        np.save(record_path, training_record)
    
    def training_and_only_saving_record(self, model_name):
        # 训练开始
        best_metric = 0
        weightedacc_list = []
        macroacc_list = []
        record_path = os.path.join(self.save_path, model_name+'_training_record.npy')
        for epoch in range(1, self.epochs + 1):
            if (epoch % 2) == 0:
                # stage 2
                self.optimizer.param_groups[0]['lr']=0.1*self.optimizer.param_groups[0]['lr']
                loss = self.train_one_epoch(epoch, 2)
                self.optimizer.param_groups[0]['lr']=10*self.optimizer.param_groups[0]['lr']
            else:
                # stage 1
                loss = self.train_one_epoch(epoch, 1)
            weightedacc, macroacc, macrof1 = cal_metrics_for_saving(epoch, self.epochs, self.model, self.valid_dataloader, self.class_num)
            weightedacc_list.append(weightedacc)
            macroacc_list.append(macroacc)
            self.scheduler.step()
        training_record = np.array([weightedacc_list, macroacc_list])
        np.save(record_path, training_record)
        

    # load and train and save
    def load_and_training_and_saving(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_metric = 0
        weightedacc_list = []
        macroacc_list = []
        model_path = os.path.join(self.save_path, self.model_name+'.pt')
        record_path = os.path.join(self.save_path, self.model_name+'_training_record.npy')
        for epoch in range(epoch+1, self.epochs + 1):
            if (epoch % 2) == 0:
                # stage 2
                self.optimizer.param_groups[0]['lr']=0.1*self.optimizer.param_groups[0]['lr']
                loss = self.train_one_epoch(epoch, 2)
                self.optimizer.param_groups[0]['lr']=10*self.optimizer.param_groups[0]['lr']
            else:
                # stage 1
                loss = self.train_one_epoch(epoch, 1)
            weightedacc, macroacc = cal_metrics_for_saving(epoch, self.epochs, self.model, self.valid_dataloader, self.class_num)
            if macroacc > best_metric:
                best_metric = macroacc
                globlist = glob.glob(os.path.join(self.save_path, self.model_name+'*'))
                if(len(globlist) == 1):
                    os.remove(globlist[0])
                print('saving best model')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, model_path)
            weightedacc_list.append(weightedacc)
            macroacc_list.append(macroacc)
            self.scheduler.step()
        training_record = np.array([weightedacc_list, macroacc_list])
        np.save(record_path, training_record)