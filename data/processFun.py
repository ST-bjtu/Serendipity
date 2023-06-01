import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
from config import max_history_len


def getEuclideanDistance(history_embedding, target_item_embedding, user_history_mask):
    batch_size, seq_len, itemEmbeddingSize = history_embedding.size()
    target_expand = target_item_embedding.expand(
        batch_size, seq_len, itemEmbeddingSize)
    distance = (history_embedding -
                target_expand).pow(2).sum(dim=2).sqrt()
    distance = distance.masked_fill(user_history_mask, 0)
    distance = distance.sum(
        dim=1) / (user_history_mask == False).sum(dim=1)
    distance = distance.unsqueeze(dim=1)

    return distance


def getDistancePointWise(datapath, itemEmbeddingPath, device, dataSetPath):
    dataset = np.load(datapath)
    dataSize = len(dataset)
    print("==========compute distance==========")
    batchSize = 32
    distance_arr = np.zeros((dataSize, 1))
    newsEmbedding = nn.Embedding.from_pretrained(
        torch.load(itemEmbeddingPath)).to(device)
    for index in tqdm(range(0, dataSize, batchSize)):
        curBatch = torch.from_numpy(
            dataset[index:index+batchSize]).to(device)
        history = curBatch[:, :max_history_len]
        target_item = curBatch[:, max_history_len]

        history_embedding = newsEmbedding(history.long())
        target_item_embedding = newsEmbedding(target_item.unsqueeze(
            1).long())

        user_history_mask = torch.where(
            history == 0, 1, 0).bool().to(device)
        target_item_dis = getEuclideanDistance(
            history_embedding, target_item_embedding, user_history_mask).cpu().numpy()

        distance_arr[index:index+batchSize] = target_item_dis

    final_dataset = np.concatenate((dataset, distance_arr), axis=-1)
    np.save(dataSetPath, final_dataset)
    print(final_dataset.shape)


def normDistance(data_path, normPath, device):
    dataset = torch.from_numpy(np.load(data_path)).to(device)
    newDataset = dataset.clone()
    unique_behavior_id = dataset[:, -2].unique()
    for behavior_id in tqdm(unique_behavior_id):
        mask = (dataset[:, -2] == behavior_id)
        cur_data_dis = dataset[:, -1][mask]
        new_data_dis = (cur_data_dis - cur_data_dis.min()) / \
            (cur_data_dis.max() - cur_data_dis.min())
        newDataset[:, -1][mask] = new_data_dis

    np.save(normPath, newDataset.cpu().numpy())
