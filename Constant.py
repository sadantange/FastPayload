from sys import platform
import torch

class Const(object):
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:  # 判断是否已经被赋值，如果是则报错
            raise self.ConstError("Can't change const.%s" % name)
        self.__dict__[name] = value

const = Const()
if platform == "win32":
    const.DEVICE = torch.device("cuda")
    const.UNSWNB15_DATA_PATH_DICT = {
        "2d_downsampled_784_6class": {
            "train": "D:\\tmp\\dataset\\UNSW-NB15\\preprocessed\\downsampled\\784_2d_6class\\train",
            "test": "D:\\tmp\\dataset\\UNSW-NB15\\preprocessed\\downsampled\\784_2d_6class\\test"
        },
        "2d_downsampled_784_10class": {
            "train": "D:\\tmp\\dataset\\UNSW-NB15\\preprocessed\\downsampled\\784_2d_10class\\train",
            "test": "D:\\tmp\\dataset\\UNSW-NB15\\preprocessed\\downsampled\\784_2d_10class\\test"
        },
        "2d_downsampled_784_9class": {
            "train": "D:\\tmp\\dataset\\UNSW-NB15\\preprocessed\\downsampled\\784_2d_9class\\train",
            "test": "D:\\tmp\\dataset\\UNSW-NB15\\preprocessed\\downsampled\\784_2d_9class\\test"
        },
        "1d_downsampled_1500_9class_v1": {
            "train": "D:\\tmp\\dataset\\UNSW-NB15\\preprocessed\\downsampled\\1500_1d_9class_v1\\train",
            "test": "D:\\tmp\\dataset\\UNSW-NB15\\preprocessed\\downsampled\\1500_1d_9class_v1\\test"
        },
        "1d_downsampled_1500_9class_v2": {
            "train": "D:\\tmp\\dataset\\UNSW-NB15\\preprocessed\\downsampled\\1500_1d_9class_v2\\train",
            "test": "D:\\tmp\\dataset\\UNSW-NB15\\preprocessed\\downsampled\\1500_1d_9class_v2\\test"
        },
        "1d_downsampled_784_9class": {
            "train": "D:\\tmp\\dataset\\UNSW-NB15\\preprocessed\\downsampled\\784_1d_9class\\train",
            "test": "D:\\tmp\\dataset\\UNSW-NB15\\preprocessed\\downsampled\\784_1d_9class\\test"
        },
        "1d_downsampled_5000_9class": {
            "train": "D:\\tmp\\dataset\\UNSW-NB15\\preprocessed\\downsampled\\5000_1d_9class\\train",
            "test": "D:\\tmp\\dataset\\UNSW-NB15\\preprocessed\\downsampled\\5000_1d_9class\\test"
        },
        "1d_downsampled_50_9class_v2": {
            "train": "D:\\tmp\\dataset\\UNSW-NB15\\preprocessed\\downsampled\\50_1d_9class_v2\\train",
            "test": "D:\\tmp\\dataset\\UNSW-NB15\\preprocessed\\downsampled\\50_1d_9class_v2\\test"
        }
    }
    const.MODEL_HYPEPARAM = {
        'epoch': 120,
        'lr': 0.001,
        'gamma': 0.5,
        'step_size': 40,
        'batch_size': 64,
    }
    const.SAVE_PATH = "E:\\tmp\\saved_models\\"
else:
    const.DEVICE = torch.device("cpu")