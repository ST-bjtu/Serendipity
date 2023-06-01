from sklearn.metrics import roc_auc_score
# from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]  
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def novelty_score(dis_list, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    dis_list = np.take(dis_list, order[:k])
    return np.mean(dis_list)


def cal_metric(predicted, labels, novelty_list):
    print("==========cal metric==========")
    # min_max_scaler = MinMaxScaler()
    all_auc = []
    auc_weight = []
    all_ndcg_5 = []
    all_ndcg_10 = []
    all_ndcg_15 = []
    all_mrr = []
    all_novelty_5 = []
    all_novelty_10 = []
    all_novelty_15 = []

    all_predicted = []
    all_labels = []
    for imp, imp_preds in tqdm(predicted.items()):
        imp_labels = labels[imp]
        imp_novelty = novelty_list[imp]
        all_predicted.extend(imp_preds)
        all_labels.extend(imp_labels)

        cur_label_sum = np.sum(imp_labels)
        if(cur_label_sum != 0) and (cur_label_sum != len(imp_labels)):
            novelty_array = np.array(imp_novelty).reshape((-1, 1))
            # novelty_array = min_max_scaler.fit_transform(novelty_array)

            mrr = mrr_score(y_true=imp_labels, y_score=imp_preds)
            auc = roc_auc_score(y_true=imp_labels, y_score=imp_preds)
            ncg_5 = ndcg_score(y_true=imp_labels, y_score=imp_preds, k=5)
            ncg_10 = ndcg_score(y_true=imp_labels, y_score=imp_preds, k=10)
            ncg_15 = ndcg_score(y_true=imp_labels, y_score=imp_preds, k=15)
            novelty_5 = novelty_score(novelty_array, imp_preds, k=5)
            novelty_10 = novelty_score(novelty_array, imp_preds, k=10)
            novelty_15 = novelty_score(novelty_array, imp_preds, k=15)
            all_auc.append(auc)
            auc_weight.append(len(imp_labels))
            all_ndcg_5.append(ncg_5)
            all_ndcg_10.append(ncg_10)
            all_ndcg_15.append(ncg_15)
            all_mrr.append(mrr)
            all_novelty_5.append(novelty_5)
            all_novelty_10.append(novelty_10)
            all_novelty_15.append(novelty_15)

    total_auc = roc_auc_score(y_true=all_labels, y_score=all_predicted)
    res = dict()
    res['ndcg@5'] = np.mean(all_ndcg_5)
    res['ndcg@10'] = np.mean(all_ndcg_10)
    res['ndcg@15'] = np.mean(all_ndcg_15)
    res['novelty@5'] = np.mean(all_novelty_5)
    res['novelty@10'] = np.mean(all_novelty_10)
    res['novelty@15'] = np.mean(all_novelty_15)
    # res["AD@10"] = (res['ndcg@10'] * res['novelty@10']) / \
    #     (res['ndcg@10'] + res['novelty@10'])
    res['auc'] = total_auc
    res['gauc'] = np.average(all_auc, weights=auc_weight)
    res['mrr'] = np.mean(all_mrr)
    
    return res
