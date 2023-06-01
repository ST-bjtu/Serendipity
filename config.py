import os
import torch

data_dir = "data/"
train_neg_sample = 4
test_neg_sample = 99
max_history_len = 50
mind_device = torch.device("cuda:2")
movie_device = torch.device("cuda:0")
amazon_device = torch.device("cuda:2")
# max_news_history_length = 50
# movie_neg_sample = 9


class BaseConfig:
    def __init__(self, args, iv=False, concat_iv=False, mediate=False, reconstruct=False, naive=False, re_true=False):
        self.iv = iv
        self.concat_iv = concat_iv
        self.mediate = mediate
        self.naive = naive
        self.reconstruct = reconstruct
        self.re_true = re_true

        self.weightedSum = mediate

        self.add_residual = False
        self.preTrainEmbed = args.preTrainEmbed
        self.train = args.train
        self.description = args.algos
        self.device = args.device
        self.dataset = args.dataset
        self.loss = args.loss
        self.epochs = args.epochs
        self.batch_size = args.batchSize
        self.start_epoch = args.startEpoch
        self.learning_rate = args.learningRate
        self.use_dis = args.distance
        self.listBPRweight = args.listW

        self.dis_loss = args.discrepencyloss
        self.dis_loss_weight = args.disW
        self.interest_novelty_loss_weight = args.inW

        self.pad_len = max_history_len
        self.print_interval = 100
        self.l2_penalty = args.l2
        self.trade_off = args.trade

        self.random_state = args.random

        if self.random_state == 2022:
            self.modelInfo = "loss_{} lr_{} device_{} l2_{} his_{}".format(
                self.loss, self.learning_rate, self.device, self.l2_penalty, self.pad_len)
        else:
            self.modelInfo = "loss_{} lr_{} device_{} l2_{} his_{} random_{}".format(
                self.loss, self.learning_rate, self.device, self.l2_penalty, self.pad_len, self.random_state)
        if self.use_dis:
            self.modelInfo = "dis " + self.modelInfo
        if self.loss == "listBPR":
            self.modelInfo += " list_{}".format(self.listBPRweight)
        if self.loss == "trade":
            self.modelInfo += " trade_{}".format(self.trade_off)
        if self.preTrainEmbed:
            self.modelInfo += " preTrain"

        if args.dataset == "MINDsmall":
            self.item_num = 65239
        elif args.dataset == "MINDlarge":
            self.item_num = 104152
        elif args.dataset == "ml-25m":
            self.item_num = 62424
        elif args.dataset == "ml-1m":
            self.item_num = 3884
        elif args.dataset == "Movies_and_TV_5":
            self.item_num = 181840
        else:
            raise ValueError("dataset error")

        if args.dataset == "MINDsmall":
            self.train_base_dir = os.path.join(data_dir, "MINDsmall_train")
            self.test_base_dir = os.path.join(data_dir, "MINDsmall_dev")
        elif args.dataset == "MINDlarge":
            self.train_base_dir = os.path.join(data_dir, "MINDlarge_train")
            self.test_base_dir = os.path.join(data_dir, "MINDlarge_dev")
        else:
            self.train_base_dir = os.path.join(data_dir, args.dataset)
            self.test_base_dir = os.path.join(data_dir, args.dataset)

        self.IVembeddingSize = 768
        self.itemEmbeddingPath = os.path.join(
            self.train_base_dir, "itemEmbedding.pt")
        self.itemIVEmbeddingPath = os.path.join(
            self.train_base_dir, "itemIVembedding.pt")

        # if self.use_dis:
        #     if self.dataset == 'ml-25m':
        #         self.train_data_path = os.path.join(
        #             self.train_base_dir, "trainset_dis_{}.npy".format(self.pad_len))
        #         self.test_data_path = os.path.join(
        #             self.test_base_dir, "testset_dis_{}.npy".format(self.pad_len))
        #     else:
        #         self.train_data_path = os.path.join(
        #             self.train_base_dir, "trainset_dis.npy")
        #         self.test_data_path = os.path.join(
        #             self.test_base_dir, "testset_dis.npy")
        # else:
        #     self.train_data_path = os.path.join(
        #         self.train_base_dir, "trainset_nov.npy")
        #     self.test_data_path = os.path.join(
        #         self.test_base_dir, "testset_nov.npy")

        self.define_params()
        self.finalPath = self.createSaveDir()
        self.csvPath = self.createCSVdir()

    def createCSVdir(self):
        csvPath = os.path.join(self.dataset, "csvResult")
        if not os.path.exists(csvPath):
            os.mkdir(csvPath)
        csvPath = os.path.join(csvPath, self.description)
        if not os.path.exists(csvPath):
            os.mkdir(csvPath)
        csvPath = os.path.join(csvPath, self.modelInfo) + ".csv"
        return csvPath

    def createSaveDir(self):
        base_model_save_path = "{}/".format(self.dataset)
        if not os.path.exists(base_model_save_path):
            os.mkdir(base_model_save_path)

        base_model_save_path = os.path.join(
            base_model_save_path, "checkpoints")
        if not os.path.exists(base_model_save_path):
            os.mkdir(base_model_save_path)

        finalPath = os.path.join(base_model_save_path, self.description)
        if not os.path.exists(finalPath):
            os.mkdir(finalPath)
        return finalPath

    def define_params(self):
        self.itemEmbeddingSize = 64
        if self.preTrainEmbed:
            self.item_mlp = {
                "hidden_dims": [self.IVembeddingSize, self.itemEmbeddingSize],
                "dropout": [0.1],
                "is_dropout": True,
            }

        if self.iv:
            self.ridge_lambd = 0.1
            self.itemIV_mlp = {
                "hidden_dims": [self.IVembeddingSize, self.itemEmbeddingSize],
                "dropout": [0.1],
                "is_dropout": True,
            }

        if self.concat_iv:
            self.ivConcat_mlp = {
                "hidden_dims": [self.itemEmbeddingSize*2, self.itemEmbeddingSize],
                "dropout": [0.1],
                "is_dropout": True,
            }

        if self.mediate:
            self.modelInfo += " disW_{} inW_{}".format(
                self.dis_loss_weight, self.interest_novelty_loss_weight)

            self.noveltyMLP = {
                "hidden_dims": [self.itemEmbeddingSize, 64, self.itemEmbeddingSize],
                "dropout": [0.1, 0.1],
                "is_dropout": True,
            }

            self.alpha_novelty_interest = {
                "hidden_dims": [self.itemEmbeddingSize*4, 128, 64, 2],
                "dropout": [0.1, 0.1, 0.1],
                "is_dropout": True,
            }

            # self.alpha_interest = {
            #     "hidden_dims": [self.itemEmbeddingSize*2, 64, 1],
            #     "dropout": [0.1, 0.1],
            #     "is_dropout": True,
            # }
            # self.alpha_novelty = {
            #     "hidden_dims": [self.itemEmbeddingSize*2, 64, 1],
            #     "dropout": [0.1, 0.1],
            #     "is_dropout": True,
            # }

        if self.naive:
            self.modelInfo += " inW_{}".format(
                self.interest_novelty_loss_weight)

            self.alpha_novelty_interest = {
                "hidden_dims": [self.itemEmbeddingSize*2, 128, 64, 2],
                "dropout": [0.1, 0.1, 0.1],
                "is_dropout": True,
            }

            # self.alpha_interest = {
            #     "hidden_dims": [self.itemEmbeddingSize, 64, 1],
            #     "dropout": [0.1, 0.1],
            #     "is_dropout": True,
            # }
            # self.alpha_novelty = {
            #     "hidden_dims": [self.itemEmbeddingSize, 64, 1],
            #     "dropout": [0.1, 0.1],
            #     "is_dropout": True,
            # }

        if self.reconstruct:
            if self.re_true:
                self.alpha_1 = {
                    "hidden_dims": [self.itemEmbeddingSize*3, 64, 1],
                    "dropout": [0.1, 0.1],
                    "is_dropout": True,
                }
                self.alpha_2 = {
                    "hidden_dims": [self.itemEmbeddingSize*3, 64, 1],
                    "dropout": [0.1, 0.1],
                    "is_dropout": True,
                }
            else:
                self.alpha_1 = {
                    "hidden_dims": [self.itemEmbeddingSize*2, 64, 1],
                    "dropout": [0.1, 0.1],
                    "is_dropout": True,
                }
                self.alpha_2 = {
                    "hidden_dims": [self.itemEmbeddingSize*2, 64, 1],
                    "dropout": [0.1, 0.1],
                    "is_dropout": True,
                }


class SASRec_config(BaseConfig):
    def __init__(self, args, iv=False, concat_iv=False, mediate=False, reconstruct=False, naive=False, re_true=False):
        super().__init__(args, iv, concat_iv, mediate, reconstruct, naive, re_true)
        self.num_layers = 1
        self.num_heads = 1
        self.dropout = 0.1


class ComiRec_config(BaseConfig):
    def __init__(self, args, iv=False, concat_iv=False, mediate=False, reconstruct=False, naive=False, re_true=False):
        super().__init__(args, iv, concat_iv, mediate, reconstruct, naive, re_true)


class Caser_config(BaseConfig):
    def __init__(self, args, iv=False, concat_iv=False, mediate=False, reconstruct=False, naive=False, re_true=False):
        super().__init__(args, iv, concat_iv, mediate, reconstruct, naive, re_true)


class GRU4Rec_config(BaseConfig):
    def __init__(self, args, iv=False, concat_iv=False, mediate=False, reconstruct=False, naive=False, re_true=False):
        super().__init__(args, iv, concat_iv, mediate, reconstruct, naive, re_true)
        self.hidden_size = 128
        self.num_layers = 1


class NRHUB_Config(BaseConfig):
    def __init__(self, args, iv=False, concat_iv=False, mediate=False, reconstruct=False, naive=False, re_true=False):
        super().__init__(args, iv, concat_iv, mediate, reconstruct, naive, re_true)
        self.Dense_dim = self.itemEmbeddingSize


class DIN_Config(BaseConfig):
    def __init__(self, args, iv=False, concat_iv=False, mediate=False, reconstruct=False, naive=False, re_true=False):
        super().__init__(args, iv, concat_iv, mediate, reconstruct, naive, re_true)


# TODO: 分析实验
class IVmediate_linear_Config(BaseConfig):
    def __init__(self, args, iv=False, concat_iv=False, mediate=False, reconstruct=False, naive=False, re_true=False):
        super().__init__(args, iv, concat_iv, mediate, reconstruct, naive, re_true)


class PURS_Config(BaseConfig):
    def __init__(self, args, iv=False, concat_iv=False, mediate=False, reconstruct=False, naive=False, re_true=False):
        super().__init__(args, iv, concat_iv, mediate, reconstruct, naive, re_true)
        self.itemEmbeddingSize = 32
        if self.preTrainEmbed:
            self.item_mlp = {
                "hidden_dims": [self.IVembeddingSize, self.itemEmbeddingSize],
                "dropout": [0.1],
                "is_dropout": True,
            }
        self.hidden_size = 64
        self.long_memory_window = 50
        self.short_memory_window = 15
        self.num_layers = 1


class DESR_Config(BaseConfig):
    def __init__(self, args, iv=False, concat_iv=False, mediate=False, reconstruct=False, naive=False, re_true=False):
        super().__init__(args, iv, concat_iv, mediate, reconstruct, naive, re_true)


class SNPR_Config(BaseConfig):
    def __init__(self, args, iv=False, concat_iv=False, mediate=False, reconstruct=False, naive=False, re_true=False):
        super().__init__(args, iv, concat_iv, mediate, reconstruct, naive, re_true)
        self.num_layers = 1
        self.num_heads = 1
        self.dropout = 0.1
        # self.trade_off = 0.7
