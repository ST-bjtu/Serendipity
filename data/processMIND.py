import json
import os
import numpy as np
import pandas as pd
import torch
from config import max_history_len, mind_device
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def generateNewsDictAndEmbeddings(train_news_path, test_news_path,
                                  newsDictPath="data/MINDsmall_train/"):
    print("==========generate newsDict==========")
    train_news_data = pd.read_table(train_news_path,
                                    header=None,
                                    names=[
                                        'id', 'category', 'subcategory', 'title', 'abstract', 'url',
                                        'title_entities', 'abstract_entities'
                                    ])
    print(train_news_data.info())
    print("trainNewsSize:{}".format(len(train_news_data)))
    test_news_data = pd.read_table(test_news_path,
                                   header=None,
                                   names=[
                                       'id', 'category', 'subcategory', 'title', 'abstract', 'url',
                                       'title_entities', 'abstract_entities'
                                   ])
    print(test_news_data.info())
    print("testNewsSize:{}".format(len(test_news_data)))
    news_data = pd.merge(train_news_data, test_news_data, how='outer')
    print(news_data.info())
    print("newsDictSize:{}".format(len(news_data)))
    pad_news = pd.DataFrame({"id": ["<pad>"], "category": [None], "subcategory": [None], "title": [None], "abstract": [None],
                             "url": [None], 'title_entities': [None], 'abstract_entities': [None]})
    news_data = pd.concat([pad_news, news_data])
    news_data.index = pd.Series(list(range(len(news_data))))
    news_data.to_pickle(os.path.join(newsDictPath, "newsDict.pkl"))

    generateNewsEmbedding(newsDictPath, news_data)


def generateNewsEmbedding(newsDictPath, news_data):
    print("==========generate newsIVembedding & newsEmbedding==========")
    model = AutoModel.from_pretrained(
        'distilbert-base-uncased').to(mind_device)
    tokenizer = AutoTokenizer.from_pretrained(
        'distilbert-base-uncased', do_lower_case=True)

    entity_null = news_data['title_entities'].isnull().to_numpy()
    abstract_null = news_data['abstract'].isnull().to_numpy()
    newsQueryList = []
    newsContentList = []
    batch_size = 8
    temp_query_batch = []
    temp_content_batch = []
    for i in tqdm(range(len(news_data))):
        if i == 0:
            continue
        category = news_data.iloc[i]['category']
        subcategory = news_data.iloc[i]['subcategory']
        title = news_data.iloc[i]['title']
        if abstract_null[i]:
            curContent = title
        else:
            abstract = news_data.iloc[i]['abstract']
            curContent = " ".join([title, abstract])
        if entity_null[i]:
            curQuery = ' '.join([category, subcategory])
        else:
            title_entities = json.loads(news_data.iloc[i]['title_entities'])
            title_entities_list = " ".join(
                [x['Label'] for x in title_entities])
            curQuery = ' '.join([category, subcategory, title_entities_list])
        temp_query_batch.append(curQuery)
        temp_content_batch.append(curContent)
        if (i + 1) % batch_size == 0:
            newsQueryList.append(temp_query_batch)
            temp_query_batch = []
            newsContentList.append(temp_content_batch)
            temp_content_batch = []
    if len(temp_query_batch) > 0:
        newsQueryList.append(temp_query_batch)
    if len(temp_content_batch) > 0:
        newsContentList.append(temp_content_batch)
    print("==========generate newsIVembedding==========")
    newsIVembedding = torch.zeros((1, model.config.hidden_size))
    for batch in tqdm(newsQueryList):
        queryEncoded = tokenizer(batch, padding="max_length",
                                 truncation=True, return_tensors="pt", max_length=100)
        queryEncoded = {k: v.to(mind_device) for k, v in queryEncoded.items()}
        outputs = model(**queryEncoded)
        embedding = outputs.last_hidden_state[:, 0, :].detach().cpu()
        newsIVembedding = embedding if newsIVembedding is None else torch.cat(
            [newsIVembedding, embedding], dim=0)
    torch.save(newsIVembedding, os.path.join(
        newsDictPath, "itemIVembedding.pt"))

    print("==========generate newsEmbedding==========")
    newsEmbedding = torch.zeros((1, model.config.hidden_size))
    for batch in tqdm(newsContentList):
        contentEncoded = tokenizer(
            batch, padding="max_length", truncation=True, return_tensors="pt", max_length=100)
        contentEncoded = {k: v.to(mind_device)
                          for k, v in contentEncoded.items()}
        outputs = model(**contentEncoded)
        embedding = outputs.last_hidden_state[:, 0, :].detach().cpu()
        newsEmbedding = embedding if newsEmbedding is None else torch.cat(
            [newsEmbedding, embedding], dim=0)
    torch.save(newsEmbedding, os.path.join(newsDictPath, "itemEmbedding.pt"))


def processData(behaviors_path, newsDictPath, dataSetPath):
    print("behaviors_path:{} newsDictPath:{}".format(
        behaviors_path, newsDictPath))
    print("==========load behaviors==========")
    behaviors_data = pd.read_table(
        behaviors_path,
        header=None,
        names=['impression_id', 'user_id', 'time', 'history', 'impressions'])

    print("==========load newsDict==========")
    news_data = pd.read_pickle(os.path.join(
        newsDictPath, "newsDict.pkl"))
    print("newsDictSize:{}".format(len(news_data)))
    id2news = news_data['id'].to_list()
    news2id = {id2news[i]: i for i in range(len(id2news))}

    print("==========generate dataset==========")
    history_null = behaviors_data['history'].isnull().to_numpy()
    dataset = []
    for behavior in tqdm(behaviors_data.itertuples()):
        idx, impression_id, user_id, time, history, impressions = behavior
        if history_null[idx]:
            continue
        history = history.split(" ")
        news_history = [news2id[x] for x in history]
        news_history = news_history[:max_history_len]

        impressions = impressions.split(" ")
        target_item_list = []
        target_label_list = []
        for imp in impressions:
            item, label = imp.split('-')
            target_item_list.append(news2id[item])
            target_label_list.append(int(label))

        if len(news_history) < max_history_len:
            news_history = news_history + \
                [0]*(max_history_len-len(news_history))

        for i in range(len(target_item_list)):
            dataset.append(
                news_history + [target_item_list[i], target_label_list[i], impression_id])
    dataset = np.array(dataset)
    print("dataSize:{}".format(len(dataset)))
    np.save(dataSetPath, dataset)
