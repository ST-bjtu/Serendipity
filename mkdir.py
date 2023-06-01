import os

# dataset = ["MINDsmall", "ml-1m", "MINDlarge", "Movies_and_TV_5", "ml-25m"]
dataset = ["MINDsmall", "ml-1m"]

# algo_list = ["din", "din_dice", "din_ivmediate", "din_ivmediate_re", "din_ivconcat", "din_naiveNovelty",
#              "NRHUB", "NRHUB_dice", "NRHUB_ivmediate", "NRHUB_ivmediate_re", "NRHUB_ivconcat", "NRHUB_naiveNovelty",
#              "SASRec", "SASRec_ivmediate_re", "SASRec_dice", "GRU4Rec_ivconcat",
#              "GRU4Rec", "GRU4Rec_ivmediate_re", "GRU4Rec_dice", "SASRec_ivconcat",
#              "PURS"]

base_model = ["din", "NRHUB", "SASRec", "GRU4Rec"]
moduls = [None, "dice", "ivmediate",
          "ivmediate_re", "ivmediate_re_true", "ivconcat", "naiveNovelty"]
baselines = ["PURS", "SNPR"]

for data in dataset:
    if not os.path.exists(data):
        os.mkdir(data)
    out_path = os.path.join(data, "output")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for base in base_model:
        for m in moduls:
            if m is None:
                algo_path = os.path.join(out_path, base)
            else:
                algo_path = os.path.join(out_path, "{}_{}".format(base, m))
            print(algo_path)
            if not os.path.exists(algo_path):
                os.mkdir(algo_path)
    for baseline in baselines:
        algo_path = os.path.join(out_path, baseline)
        print(algo_path)
        if not os.path.exists(algo_path):
            os.mkdir(algo_path)
