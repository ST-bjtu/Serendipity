import torch
import numpy as np
from torch.utils.data.dataset import IterableDataset
from torch.utils.data import DataLoader
from config import max_history_len


class trainPointBatchData:
    def __init__(self, datapath, batch_size):
        self.dataset = np.load(datapath)
        self.dataset_iter = iter(self.dataset)
        self.batch_size = batch_size
        print("==========datasize:{}==========".format(self.dataset.shape))

        self.max_history_length = max_history_len
        self.batch = None
        self.batch_count = 0
        self.preID = -1
        self.iter_end = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_end:
            self.dataset_iter = iter(self.dataset)
            self.iter_end = False
            self.batch = None
            self.batch_count = 0
            self.preID = -1
            raise StopIteration
        x = self.max_history_length
        while True:
            try:
                curData = next(self.dataset_iter)
                behavior_id = curData[-2]
                if behavior_id == self.preID:
                    self.batch = curData.reshape((1, -1)) if self.batch is None else np.concatenate(
                        (self.batch, curData.reshape((1, -1))), axis=0)
                    self.batch_count = len(self.batch)
                else:
                    if self.batch_count >= self.batch_size:
                        batch_tensor = torch.tensor(self.batch)
                        self.batch = curData.reshape((1, -1))
                        self.batch_count = len(self.batch)
                        break
                    else:
                        self.batch = curData.reshape((1, -1)) if self.batch is None else np.concatenate(
                            (self.batch, curData.reshape((1, -1))), axis=0)
                        self.batch_count = len(self.batch)
                self.preID = behavior_id
            except StopIteration:
                batch_tensor = torch.tensor(self.batch)
                self.iter_end = True
                break

        return (batch_tensor[:, :x], batch_tensor[:, x], batch_tensor[:, x+1], batch_tensor[:, x+2], batch_tensor[:, x+3])


class testPointWiseDataSet(IterableDataset):
    def __init__(self, dataName) -> None:
        super().__init__()
        if dataName == "MINDsmall":
            self.dataset = np.load("data/MINDsmall_dev/testset_dis_norm.npy")
        elif dataName == "MINDlarge":
            self.dataset = np.load("data/MINDlarge_dev/testset_dis_norm.npy")
        else:
            self.dataset = np.load(
                "data/{}/valset_dis_norm.npy".format(dataName))
        self.datasize = len(self.dataset)
        print("==========datasize:{}==========".format(self.dataset.shape))

    def __len__(self):
        return self.datasize

    def __iter__(self):
        for row in self.dataset:
            yield row[:max_history_len], row[max_history_len], row[max_history_len+1], row[max_history_len+2], row[max_history_len+3]


def getDataLoader(train, batchSize, dataName):
    if train:
        if dataName == "MINDsmall":
            datapath = "data/MINDsmall_train/trainset_dis_norm.npy"
        elif dataName == "MINDlarge":
            datapath = "data/MINDlarge_train/trainset_dis_norm.npy"
        else:
            datapath = "data/{}/trainset_dis_norm.npy".format(dataName)
        return trainPointBatchData(datapath, batchSize)
    else:
        dataset = testPointWiseDataSet(dataName)
        dataloader = DataLoader(dataset, batch_size=batchSize,
                                pin_memory=True, shuffle=False)
        return dataloader
