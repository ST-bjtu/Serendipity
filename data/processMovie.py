import os
import random
import numpy as np
import pandas as pd
import torch
from config import max_history_len, movie_device, test_neg_sample, train_neg_sample
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def generateEmbeddings(movies, data_path):
    print("==========generate movieIVembedding & movieEmbedding==========")
    model = AutoModel.from_pretrained(
        'distilbert-base-uncased').to(movie_device)
    tokenizer = AutoTokenizer.from_pretrained(
        'distilbert-base-uncased', do_lower_case=True)

    genres_null = movies['genres'].isnull().to_numpy()
    title_null = movies['title'].isnull().to_numpy()
    print("genres_null:{} title_null:{}".format(
        genres_null.sum(), title_null.sum()))

    batch_size = 64
    movieQueryList = []
    movieContentList = []
    temp_query_batch = []
    temp_content_batch = []
    for movie_data in tqdm(movies.itertuples()):
        idx, movie_id, title, genres = movie_data
        if idx == 0:
            continue
        temp_content_batch.append(title)
        temp_query_batch.append(" ".join(genres.split("|")))
        if (idx+1) % batch_size == 0:
            movieQueryList.append(temp_query_batch)
            movieContentList.append(temp_content_batch)
            temp_content_batch = []
            temp_query_batch = []
    if len(temp_query_batch) > 0:
        movieQueryList.append(temp_query_batch)
    if len(temp_content_batch) > 0:
        movieContentList.append(temp_content_batch)

    print("==========generate movieIVembedding==========")
    movieIVembedding = torch.zeros((1, model.config.hidden_size))
    for batch in tqdm(movieQueryList):
        queryEncoded = tokenizer(batch, padding="max_length",
                                 truncation=True, return_tensors="pt", max_length=100)
        queryEncoded = {k: v.to(movie_device) for k, v in queryEncoded.items()}
        outputs = model(**queryEncoded)
        embedding = outputs.last_hidden_state[:, 0, :].detach().cpu()
        movieIVembedding = embedding if movieIVembedding is None else torch.cat(
            [movieIVembedding, embedding], dim=0)
    torch.save(movieIVembedding, os.path.join(
        data_path, "itemIVembedding.pt"))
    print("movieIVembedding:{}".format(movieIVembedding.size()))

    print("==========generate movieEmbedding==========")
    movieEmbedding = torch.zeros((1, model.config.hidden_size))
    for batch in tqdm(movieContentList):
        contentEncoded = tokenizer(
            batch, padding="max_length", truncation=True, return_tensors="pt", max_length=100)
        contentEncoded = {k: v.to(movie_device)
                          for k, v in contentEncoded.items()}
        outputs = model(**contentEncoded)
        embedding = outputs.last_hidden_state[:, 0, :].detach().cpu()
        movieEmbedding = embedding if movieEmbedding is None else torch.cat(
            [movieEmbedding, embedding], dim=0)
    torch.save(movieEmbedding, os.path.join(
        data_path, "itemEmbedding.pt"))
    print("movieEmbedding:{}".format(movieEmbedding.size()))


def processMovie(data_path, result_path):
    print("==========max_his_len:{}==========".format(max_history_len))

    print("==========load movie dict==========")
    movies = pd.read_csv(os.path.join(data_path, "movies.csv"))
    pad_movie = pd.DataFrame(
        {"movieId": ["<pad>"], "title": [""], "genres": [""]})
    movies = pd.concat([pad_movie, movies])
    movies.index = pd.Series(list(range(len(movies))))
    id2movie = movies['movieId'].tolist()
    movie2id = {value: index for index, value in enumerate(id2movie)}

    generateEmbeddings(movies, result_path)

    print("==========load rating data==========")
    data = pd.read_csv(
        os.path.join(data_path, "ratings.csv")).drop(columns=['rating'])
    data = data.sort_values(by=['userId', 'timestamp']).drop(
        columns=['timestamp'])
    data['userId'] = data['userId'].astype('category')
    data['userId'] = data['userId'].cat.codes.values
    data['movieId'] = data['movieId'].apply(lambda x: movie2id[x])
    data.index = pd.Series(list(range(len(data))))

    dataSize = len(data)
    numUser = len(data['userId'].unique())
    numItem = len(movies)
    print("==========userNum:{} itemNum:{}==========".format(numUser, numItem))

    print("==========generate train index==========")

    def getTest(df):
        df['train'].iloc[-1] = False
        df['train'].iloc[-2] = False
        df['val'].iloc[-2] = True
        df['test'].iloc[-1] = True
        return df
    data['train'] = True
    data['val'] = False
    data['test'] = False
    data = data.groupby("userId").apply(getTest)
    trainSize = data['train'].sum()
    valSize = data['val'].sum()
    testSize = data['test'].sum()
    print("==========trainSize:{} valSize:{} testSize:{}==========".format(
        trainSize, valSize, testSize))

    userTrainSet = data[data['train']].groupby("userId").apply(
        lambda df: list(set(df['movieId']))).to_dict()
    userValSet = data[data['val']].groupby("userId").apply(
        lambda df: list(set(df['movieId']))).to_dict()
    userTestSet = data[data['test']].groupby("userId").apply(
        lambda df: list(set(df['movieId']))).to_dict()

    seqTrainSet = []
    seqValSet = []
    seqTestSet = []
    userHistory = []
    print("==========process data==========")
    for cur_data in tqdm(data.itertuples()):
        idx, userId, movieId, train, val, test = cur_data

        if len(userHistory) == 0:
            userHistory.append(movieId)
            continue

        user_behaviors = userHistory.copy(
        ) + [0] * (max_history_len - len(userHistory))

        if len(userHistory) == max_history_len:
            userHistory.pop(0)
        userHistory.append(movieId)

        cur_userTrainSee = userTrainSet[userId]
        cur_userValSee = userValSet[userId]
        cur_userTestSee = userTestSet[userId]
        if train:
            seqTrainSet.append(user_behaviors + [movieId, 1, userId])
            neg_sample_list = []
            for _ in range(train_neg_sample):
                neg_sample = random.randint(1, numItem-1)
                while (neg_sample in cur_userTrainSee) or (neg_sample in neg_sample_list):
                    neg_sample = random.randint(1, numItem-1)
                neg_sample_list.append(neg_sample)
                seqTrainSet.append(
                    user_behaviors + [neg_sample, 0, userId])
        elif val:
            seqValSet.append(user_behaviors + [movieId, 1, userId])
            neg_sample_list = []
            for _ in range(test_neg_sample):
                neg_sample = random.randint(1, numItem-1)
                while (neg_sample in cur_userTrainSee) or (neg_sample in cur_userValSee) or (neg_sample in neg_sample_list):
                    neg_sample = random.randint(1, numItem-1)
                neg_sample_list.append(neg_sample)
                seqValSet.append(user_behaviors + [neg_sample, 0, userId])
        elif test:
            seqTestSet.append(user_behaviors + [movieId, 1, userId])
            neg_sample_list = []
            for _ in range(test_neg_sample):
                neg_sample = random.randint(1, numItem-1)
                while (neg_sample in cur_userTrainSee) or (neg_sample in cur_userValSee) or (neg_sample in cur_userTestSee) or (neg_sample in neg_sample_list):
                    neg_sample = random.randint(1, numItem-1)
                neg_sample_list.append(neg_sample)
                seqTestSet.append(user_behaviors + [neg_sample, 0, userId])

        if (idx < dataSize-1) and (data['train'].iloc[idx+1]) and (not train):
            userHistory = []

    seqTrainMatrix = np.array(seqTrainSet)
    seqValMatrix = np.array(seqValSet)
    seqTestMatrix = np.array(seqTestSet)
    print("seqTrainSize: ", seqTrainMatrix.shape[0])
    print("seqValSize: ", seqValMatrix.shape[0])
    print("seqTestSize: ", seqTestMatrix.shape[0])

    np.save(os.path.join(result_path, "trainset.npy"), seqTrainMatrix)
    np.save(os.path.join(result_path, "valset.npy"), seqValMatrix)
    np.save(os.path.join(result_path, "testset.npy"), seqTestMatrix)
