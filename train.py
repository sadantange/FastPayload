import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import sys
sys.path.append(r"D:\\tmp\\IDS\\Models")
from Constant import const
# from Models.CNN2DDeepTraffic import IDCNN
# from Models.CNN1DDeepTraffic import IDCNN
# from Models.CNN1DDeepPacket import IDCNN
from Models.FastPayLoad import IDCNN
# from Models.FastPayLoadBN import IDCNN
# from Models.FastPayLoadIRB import IDCNN
# from Models.DeepFingerprint import DF
# from Models.TSCRNN import TSCRNN
# from Models.SAM import SAM
from Trainer.trainer import TwoStageTrainer
from Trainer.trainer import Trainer
from Models.loss import GHMC_Two_Stage_Loss
from Models.loss import FocalLoss
from Models.loss import Focal_Two_Stage_Loss
from Models.loss import CBTerm
from Models.loss import GHMC_Loss
from Models.loss import CE_Two_Stage_Loss
from Models.loss import LDAMLoss
from DataLoaders.UNSWNB15DataLoader import UNSWNB15DataLoader
from Models.loss import cal_class_weights_two
from Models.loss import cal_class_weights_one
class_num, class_num_list, train_dataloader, test_dataloader = UNSWNB15DataLoader(const.MODEL_HYPEPARAM['batch_size'], const.UNSWNB15_DATA_PATH_DICT["1d_downsampled_1500_9class"], mode="pro")
model = IDCNN(class_num=class_num)
criterion = GHMC_Two_Stage_Loss(weight=cal_class_weights_one(class_num_list))
optimizer = optim.AdamW(model.parameters(), lr=const.MODEL_HYPEPARAM['lr'])
scheduler = StepLR(optimizer, step_size=const.MODEL_HYPEPARAM['step_size'],
                       gamma=const.MODEL_HYPEPARAM['gamma'])
trainer = TwoStageTrainer(
    model=model, 
    criterion=criterion, 
    optimizer=optimizer, 
    scheduler=scheduler,
    train_dataloader= train_dataloader, 
    valid_dataloader= test_dataloader, 
    class_num=class_num, 
    epochs=const.MODEL_HYPEPARAM['epoch']
)
trainer.run_training_and_saving(model_name="FastPyload_epoch120")

# class_num, class_num_list, train_dataloader, test_dataloader = UNSWNB15DataLoader(const.MODEL_HYPEPARAM['batch_size'], const.UNSWNB15_DATA_PATH_DICT["1d_downsampled_1500_9class"], mode="exp")
# model = IDCNN(class_num=class_num)
# criterion = nn.CrossEntropyLoss(weight=cal_class_weights_one(class_num_list))
# optimizer = optim.AdamW(model.parameters(), lr=const.MODEL_HYPEPARAM['lr'])
# scheduler = StepLR(optimizer, step_size=const.MODEL_HYPEPARAM['step_size'],
#                        gamma=const.MODEL_HYPEPARAM['gamma'])
# trainer = Trainer(
#     model=model, 
#     criterion=criterion, 
#     optimizer=optimizer, 
#     scheduler=scheduler, 
#     train_dataloader=train_dataloader, 
#     valid_dataloader=test_dataloader, 
#     class_num=class_num, 
#     epochs=const.MODEL_HYPEPARAM['epoch'], 
#     save_path=const.SAVE_PATH
# )
# trainer.run_training()