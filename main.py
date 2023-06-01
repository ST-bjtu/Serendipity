import argparse
import os
import random

import numpy as np
import torch

from config import *
from data.dataloader import getDataLoader
from models import DIN, NRHUB, SASRec, GRU4Rec, PURS, SNPR


def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--distance", action="store_true")
parser.add_argument("--preTrainEmbed", action="store_true")
parser.add_argument("--algos", type=str, default="SNPR")
parser.add_argument("--device", type=str, default="cuda:3")
parser.add_argument("--dataset", type=str, default="MINDsmall")
parser.add_argument("--loss", type=str, default="BPR")
parser.add_argument("--discrepencyloss", type=str, default='L2')
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--bestEpoch", type=int, default=18)
parser.add_argument("--batchSize", type=int, default=5)
parser.add_argument("--startEpoch", type=int, default=0)
parser.add_argument("--hisLen", type=int, default=50)
parser.add_argument("--random", type=int, default=0)

parser.add_argument("--learningRate", type=float, default=1e-3)
parser.add_argument("--disW", type=float, default=0.0)  # dis_loss_weight
# interest_novelty_loss_weight
parser.add_argument("--inW", type=float, default=0.001)
parser.add_argument("--listW", type=float, default=1.0)
parser.add_argument("--l2", type=float, default=0.0)
parser.add_argument("--trade", type=float, default=0.7)
args = parser.parse_args()

# setup_seed(2022)
setup_seed(args.random)

# for debug
# args.train = True
# args.distance = True

print(args)
print(args.algos)
os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[1]
print("device:", os.environ["CUDA_VISIBLE_DEVICES"])

if args.algos == "din":
    config = DIN_Config(args)
    model = DIN.DeepInterestNetwork(config)
elif args.algos == "din_naiveNovelty":
    config = DIN_Config(args, naive=True)
    model = DIN.DIN_naiveNovelty(config)
elif args.algos == "din_ivconcat":
    config = DIN_Config(args, iv=True, concat_iv=True, mediate=True)
    model = DIN.DIN_IVconcat(config)
elif args.algos == "din_ivmediate":
    config = DIN_Config(args, iv=True, mediate=True, reconstruct=False)
    model = DIN.DIN_IVmediate(config)
elif args.algos == "din_ivmediate_re":
    config = DIN_Config(args, iv=True, mediate=True, reconstruct=True)
    model = DIN.DIN_IVmediate_re(config)
elif args.algos == "din_ivmediate_re_true":
    config = DIN_Config(args, iv=True, mediate=True, reconstruct=True,re_true=True)
    model = DIN.DIN_IVmediate_re_true(config)
elif args.algos == "din_dice":
    config = DIN_Config(args, iv=False, mediate=True, reconstruct=False)
    model = DIN.DIN_DICE(config)
elif args.algos == "NRHUB":
    config = NRHUB_Config(args)
    model = NRHUB.NRHUB(config)
elif args.algos == "NRHUB_naiveNovelty":
    config = NRHUB_Config(args, naive=True)
    model = NRHUB.NRHUB_naiveNovelty(config)
elif args.algos == "NRHUB_dice":
    config = NRHUB_Config(args, iv=False, mediate=True, reconstruct=False)
    model = NRHUB.NRHUB_DICE(config)
elif args.algos == "NRHUB_ivconcat":
    config = NRHUB_Config(args, iv=True, concat_iv=True, mediate=True)
    model = NRHUB.NRHUB_IVconcat(config)
elif args.algos == "NRHUB_ivmediate":
    config = NRHUB_Config(args, iv=True, mediate=True, reconstruct=False)
    model = NRHUB.NRHUB_IVmediate(config)
elif args.algos == "NRHUB_ivmediate_re":
    config = NRHUB_Config(args, iv=True, mediate=True, reconstruct=True)
    model = NRHUB.NRHUB_IVmediate_re(config)
elif args.algos == "NRHUB_ivmediate_re_true":
    config = NRHUB_Config(args, iv=True, mediate=True, reconstruct=True, re_true=True)
    model = NRHUB.NRHUB_IVmediate_re_true(config)
elif args.algos == "SASRec":
    config = SASRec_config(args)
    model = SASRec.SASRec(config)
elif args.algos == "SASRec_naiveNovelty":
    config = SASRec_config(args, naive=True)
    model = SASRec.SASRec_naiveNovelty(config)
elif args.algos == "SASRec_ivconcat":
    config = SASRec_config(args, iv=True, concat_iv=True, mediate=True)
    model = SASRec.SASRec_IVconcat(config)
elif args.algos == "SASRec_dice":
    config = SASRec_config(args, iv=False, mediate=True, reconstruct=False)
    model = SASRec.SASRec_DICE(config)
elif args.algos == "SASRec_ivmediate_re":
    config = SASRec_config(args, iv=True, mediate=True, reconstruct=True)
    model = SASRec.SASRec_IVmediate_re(config)
elif args.algos == "SASRec_ivmediate_re_true":
    config = SASRec_config(args, iv=True, mediate=True, reconstruct=True, re_true=True)
    model = SASRec.SASRec_IVmediate_re_true(config)
elif args.algos == "GRU4Rec":
    config = GRU4Rec_config(args)
    model = GRU4Rec.GRU4Rec(config)
elif args.algos == "GRU4Rec_naiveNovelty":
    config = GRU4Rec_config(args, naive=True)
    model = GRU4Rec.GRU4Rec_naiveNovelty(config)
elif args.algos == "GRU4Rec_ivconcat":
    config = GRU4Rec_config(args, iv=True, concat_iv=True, mediate=True)
    model = GRU4Rec.GRU4Rec_IVconcat(config)
elif args.algos == "GRU4Rec_dice":
    config = GRU4Rec_config(args, iv=False, mediate=True, reconstruct=False)
    model = GRU4Rec.GRU4Rec_DICE(config)
elif args.algos == "GRU4Rec_ivmediate_re":
    config = GRU4Rec_config(args, iv=True, mediate=True, reconstruct=True)
    model = GRU4Rec.GRU4Rec_IVmediate_re(config)
elif args.algos == "GRU4Rec_ivmediate_re_true":
    config = GRU4Rec_config(args, iv=True, mediate=True, reconstruct=True, re_true=True)
    model = GRU4Rec.GRU4Rec_IVmediate_re_true(config)
elif args.algos == "PURS":
    config = PURS_Config(args)
    model = PURS.PURS(config)
elif args.algos == "SNPR":
    config = SNPR_Config(args)
    model = SNPR.SNPR(config)
else:
    raise ValueError("algo error!")


print("load testSet")
test_dataloader = getDataLoader(
    train=False, batchSize=config.batch_size, dataName=config.dataset)

if args.train == True:
    print("load trainSet")
    train_dataloader = getDataLoader(
        train=True, batchSize=config.batch_size, dataName=config.dataset)
    model.fit(train_dataloader, test_dataloader)
else:
    # print("==========evaluate trainset==========")
    # model.evaluate(train_dataloader)
    print("==========evaluate testset==========")
    model.evaluate(test_dataloader)
    # model.testCurEpoch(test_dataloader, args.bestEpoch)
