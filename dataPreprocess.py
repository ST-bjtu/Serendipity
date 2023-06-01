from data.processFun import getDistancePointWise, normDistance

# # MIND small
# from data.processMIND import generateNewsDictAndEmbeddings, processData
from config import mind_device
# generateNewsDictAndEmbeddings("data/MINDsmall_train/news.tsv",
#                               "data/MINDsmall_dev/news.tsv",
#                               "data/MINDsmall_train/")
# processData("data/MINDsmall_train/behaviors.tsv", "data/MINDsmall_train/",
#             "data/MINDsmall_train/trainset.npy")
# getDistancePointWise("data/MINDsmall_train/trainset.npy", "data/MINDsmall_train/itemEmbedding.pt",
#                       mind_device, "data/MINDsmall_train/trainset_dis.npy")

# processData("data/MINDsmall_dev/behaviors.tsv", "data/MINDsmall_train/",
#             "data/MINDsmall_dev/testset.npy")
# getDistancePointWise("data/MINDsmall_dev/testset.npy", "data/MINDsmall_train/itemEmbedding.pt",
#                       mind_device, "data/MINDsmall_dev/testset_dis.npy")
normDistance("data/MINDsmall_train/trainset_dis.npy",
             "data/MINDsmall_train/trainset_dis_norm.npy", mind_device)
normDistance("data/MINDsmall_dev/testset_dis.npy",
             "data/MINDsmall_dev/testset_dis_norm.npy", mind_device)


# MIND large
# generateNewsDictAndEmbeddings("data/MINDlarge_train/news.tsv",
#                               "data/MINDlarge_dev/news.tsv",
#                               "data/MINDlarge_train/behaviors.tsv",
#                               "data/MINDlarge_train/")

# processData("data/MINDlarge_train/behaviors.tsv", "data/MINDlarge_train/",
#             "data/MINDlarge_train/trainset.npy", ifSort=False)
# processData("data/MINDlarge_dev/behaviors.tsv", "data/MINDlarge_train/",
#             "data/MINDlarge_dev/testset.npy", ifSort=False)

# getDistance("data/MINDlarge_train/newsEmbedding_100.pt", "data/MINDlarge_train/trainset.npy",
#             "data/MINDlarge_train/trainset.pt")


# getDistance("data/MINDlarge_train/newsEmbedding_100.pt", "data/MINDlarge_dev/testset.npy",
#             "data/MINDlarge_dev/testset.pt")

# ml-1m
from config import movie_device
from data.processMovie import processMovie
movieType = "ml-1m"
# processMovie("data/{}".format(movieType), "data/{}".format(movieType))
# getDistancePointWise("data/{}/trainset.npy".format(movieType), "data/{}/itemEmbedding.pt".format(movieType),
#                      movie_device, "data/{}/trainset_dis.npy".format(movieType))
# getDistancePointWise("data/{}/valset.npy".format(movieType), "data/{}/itemEmbedding.pt".format(movieType),
#                      movie_device, "data/{}/valset_dis.npy".format(movieType))
# getDistancePointWise("data/{}/testset.npy".format(movieType), "data/{}/itemEmbedding.pt".format(movieType),
#                      movie_device, "data/{}/testset_dis.npy".format(movieType))
# normDistance("data/{}/trainset_dis.npy".format(movieType),
#              "data/{}/trainset_dis_norm.npy".format(movieType), movie_device)
# normDistance("data/{}/valset_dis.npy".format(movieType),
#              "data/{}/valset_dis_norm.npy".format(movieType), movie_device)
# normDistance("data/{}/testset_dis.npy".format(movieType),
#              "data/{}/testset_dis_norm.npy".format(movieType), movie_device)

# ml-25m
# processML25m(max_his_len=50)

# getDistance("data/ml-25m/movieEmbedding.pt",
#             "data/ml-25m/trainset_50.npy", "data/ml-25m/trainset_dis_50.npy", "ml-25m", 50)

# getDistance("data/ml-25m/movieEmbedding.pt",
#             "data/ml-25m/testset_50.npy", "data/ml-25m/testset_dis_50.npy", "ml-25m", 50)
