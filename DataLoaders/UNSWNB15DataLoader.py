import torch.utils.data as torchdata
import sys
sys.path.append(r"D:\\tmp\\IDS\\Constant")
import numpy as np
import glob

class UNSWNB15Dataset(torchdata.Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.data_x, self.data_y = self.read(file_path)
        self.size = len(self.data_x)
        class_list, self.class_num_list = np.unique(self.data_y, return_counts=True)
        self.class_num = len(class_list)

    def __len__(self):
        return self.size
    
    def read(self, file_path):
        file_list = glob.glob(file_path + "*.npy")
        data_x = []
        data_y = []
        for file in file_list:
            print(file)
            data = np.load(file, allow_pickle=True)
            data_x += list(data[0])
            data_y += list(data[1])
        return data_x, data_y

    def image_normalize(self, image):
        return ((image-np.min(image)) / (np.max(image)-np.min(image))).astype(np.float32)

    def __getitem__(self, idx):
        inp_data = self.image_normalize(self.data_x[idx])
        inp_data = inp_data.astype(np.float32)
        label = self.data_y[idx]
        return inp_data, label

# mode: "exp", "pro", "eval"
def UNSWNB15DataLoader(batch_size, file_path_dic, mode):
    if mode == "exp":
        train_dataset = UNSWNB15Dataset(file_path_dic["test"])
        test_dataset = UNSWNB15Dataset(file_path_dic["test"])
        train_dataloader = torchdata.DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = torchdata.DataLoader(test_dataset, batch_size=batch_size)
        return train_dataset.class_num, train_dataset.class_num_list, train_dataloader, test_dataloader
    elif mode == "pro":
        train_dataset = UNSWNB15Dataset(file_path_dic["train"])
        test_dataset = UNSWNB15Dataset(file_path_dic["test"])
        train_dataloader = torchdata.DataLoader(train_dataset, batch_size=batch_size, drop_last=False)
        test_dataloader = torchdata.DataLoader(test_dataset, batch_size=batch_size, drop_last=False)
        return train_dataset.class_num, train_dataset.class_num_list, train_dataloader, test_dataloader
    elif mode == "eval":
        test_dataset = UNSWNB15Dataset(file_path_dic["test"])
        test_dataloader = torchdata.DataLoader(test_dataset, batch_size=batch_size, drop_last=True)
        return test_dataset.class_num, test_dataloader
        