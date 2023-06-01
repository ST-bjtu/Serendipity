import os

import torch
import torch.nn as nn
from config import *
from tqdm import tqdm
from utils.metric import cal_metric
import pandas as pd


class BasicModel(nn.Module):
    def __init__(self, config: BaseConfig):
        """
        basic model:
            create model and optimizer
            initialize model and hyper-parameter
        """
        super().__init__()
        self.device = "cuda:0"
        self.name = config.description
        self.config = config

    def _run_train(self, dataloader, optimizer, cur_epoch):

        print("==========epoch:{}==========".format(cur_epoch))
        epoch_loss = 0
        epoch_click_loss = 0
        epoch_interest_novelty_loss = 0
        epoch_discrepency_loss = 0
        tqdm_ = tqdm(iterable=dataloader)

        for step, batch in enumerate(tqdm_):
            history, target_item, label, behavior_id, novelty = batch

            num_label = len(label)
            label_sum = label.sum()
            if (label_sum == num_label) or (label_sum == 0):
                continue

            pad_mask = (target_item == 0)
            if pad_mask.sum() > 0:
                unpad_mask = (target_item != 0)
                history = history[unpad_mask]
                target_item = target_item[unpad_mask]
                label = label[unpad_mask]
                novelty = novelty[unpad_mask]
                behavior_id = behavior_id[unpad_mask]

            out_list, discrepency_loss = self(history, target_item, novelty)

            click_loss, interest_loss = self.loss_fn(out_list, label.unsqueeze(1).to(self.device), novelty.unsqueeze(1).to(
                self.device), behavior_id.unsqueeze(1).to(self.device))

            discrepency_loss *= self.config.dis_loss_weight
            loss = click_loss + interest_loss + discrepency_loss

            epoch_loss += loss.item()
            epoch_click_loss += click_loss.item()
            epoch_interest_novelty_loss += interest_loss.item()
            epoch_discrepency_loss += discrepency_loss.item()

            optimizer.zero_grad()

            if self.config.description == "SNPR":
                torch.use_deterministic_algorithms(False)
                loss.backward()
                torch.use_deterministic_algorithms(True)
            else:
                loss.backward()
            optimizer.step()

            if step % self.config.print_interval == 0 and step > 0:
                tqdm_.set_description(
                    "epoch:{:d} step:{:d} loss:{:.4f} clickLoss:{:.4f} interestLoss:{:.4f} discrepencyLoss:{:.4f}".format(
                        cur_epoch, step, epoch_loss / step, epoch_click_loss /
                        step, epoch_interest_novelty_loss / step, epoch_discrepency_loss / step
                    ))
        modelInfo = self.config.modelInfo + \
            " " + "epoch_{}.pth".format(cur_epoch)
        model_save_path = os.path.join(self.config.finalPath, modelInfo)
        print("save model to:{}".format(model_save_path))
        torch.save(self.state_dict(), model_save_path)

    def fit(self, trainDataLoader, testDataLoader):
        print("==========model training==========")
        self.train()

        bias_list = (param for name, param in self.named_parameters()
                     if name[-4:] == 'bias')
        others_list = (
            param for name, param in self.named_parameters() if name[-4:] != 'bias')
        parameters = [{'params': bias_list, 'weight_decay': 0},
                      {'params': others_list, 'weight_decay': self.config.l2_penalty}]
        optimizer = torch.optim.AdamW(
            parameters, lr=self.config.learning_rate, weight_decay=self.config.l2_penalty)
        print("==========L2 regularization:{}==========".format(
            self.config.l2_penalty))

        if self.config.start_epoch > 0:
            modelInfo = self.config.modelInfo + \
                " " + "epoch_{}.pth".format(self.config.start_epoch - 1)
            print("==========load_model_from :{}==========".format(modelInfo))
            self.load_state_dict(torch.load(os.path.join(
                self.config.finalPath, modelInfo), map_location=self.device))

        # self.evaluate(testDataLoader)

        best_result = None
        best_epoch = None
        all_result = []
        for epoch in range(self.config.start_epoch, self.config.start_epoch + self.config.epochs):
            self.train()
            self._run_train(trainDataLoader, optimizer, epoch)
            self.eval()
            preds, labels, novelty_list = self._run_eval(testDataLoader)

            # calculate metrics
            res = cal_metric(preds, labels, novelty_list)
            res['epoch'] = epoch
            print(res)

            if best_result is None:
                best_result = res
                best_epoch = epoch
            else:
                if res['gauc'] > best_result['gauc']:
                    best_result = res
                    best_epoch = epoch
                # if res['ndcg@10'] > best_result['ndcg@10']:
                #     best_result = res
                #     best_epoch = epoch
                # if res['AD@10'] > best_result['AD@10']:
                #     best_result = res
                #     best_epoch = epoch
            all_result.append(res)

        print("==========best epoch:{}==========".format(best_epoch))
        print(best_result)

        all_result_csv = pd.DataFrame(all_result).set_index("epoch")
        all_result_csv["AD@5"] = (all_result_csv['ndcg@5'] * all_result_csv['novelty@5']) / \
            (all_result_csv['ndcg@5'] + all_result_csv['novelty@5'])
        all_result_csv["AD@10"] = (all_result_csv['ndcg@10'] * all_result_csv['novelty@10']) / \
            (all_result_csv['ndcg@10'] + all_result_csv['novelty@10'])
        all_result_csv["AD@15"] = (all_result_csv['ndcg@15'] * all_result_csv['novelty@15']) / \
            (all_result_csv['ndcg@15'] + all_result_csv['novelty@15'])
        all_result_csv.to_csv(self.config.csvPath)

    def _run_eval(self, dataloader):
        all_imp_preds = {}
        all_imp_labels = {}
        all_imp_novelty = {}
        for batch in tqdm(dataloader):
            history, target_item, label, impression_id, novelty = batch
            out_list, discrepency_loss = self(history, target_item, novelty)
            pred = self.config.trade_off * out_list[0] + (1 - self.config.trade_off) * \
                novelty.unsqueeze(1).to(self.device)
            pred = pred.squeeze().tolist()
            label = label.tolist()
            impression_id = impression_id.tolist()
            novelty = novelty.tolist()

            for index, imp in enumerate(impression_id):
                cur_imp_id = int(imp)
                if cur_imp_id not in all_imp_preds.keys():
                    all_imp_preds[cur_imp_id] = []
                    all_imp_labels[cur_imp_id] = []
                    all_imp_novelty[cur_imp_id] = []
                all_imp_preds[cur_imp_id].append(pred[index])
                all_imp_labels[cur_imp_id].append(label[index])
                all_imp_novelty[cur_imp_id].append(novelty[index])

        return all_imp_preds, all_imp_labels, all_imp_novelty

    def testCurEpoch(self, dataloader, target_epoch):
        self.eval()

        print("==========target_epoch:{}==========".format(target_epoch))
        modelInfo = self.config.modelInfo + " " + \
            "epoch_{}.pth".format(target_epoch)
        print("==========load_model_from :{}==========".format(modelInfo))
        self.load_state_dict(torch.load(os.path.join(
            self.config.finalPath, modelInfo), map_location=self.device))

        preds, labels, novelty_list = self._run_eval(dataloader)

        # calculate metrics
        res = cal_metric(preds, labels, novelty_list)
        res['epoch'] = target_epoch
        print(res)
        return res

    def evaluate(self, dataloader):
        print("==========model testing==========")
        self.eval()

        best_result = None
        best_epoch = None

        all_result = []
        for epoch in range(self.config.start_epoch, self.config.start_epoch + self.config.epochs):
            res = self.testCurEpoch(dataloader, epoch)
            all_result.append(res)

            if best_result is None:
                best_result = res
                best_epoch = epoch
            else:
                if res['gauc'] > best_result['gauc']:
                    best_result = res
                    best_epoch = epoch
                # if res['ndcg@10'] > best_result['ndcg@10']:
                #     best_result = res
                #     best_epoch = epoch
                # if res['AD@10'] > best_result['AD@10']:
                #     best_result = res
                #     best_epoch = epoch

        print("==========best epoch:{}==========".format(best_epoch))
        print(best_result)

        all_result_csv = pd.DataFrame(all_result).set_index("epoch")
        all_result_csv["AD@5"] = (all_result_csv['ndcg@5'] * all_result_csv['novelty@5']) / \
            (all_result_csv['ndcg@5'] + all_result_csv['novelty@5'])
        all_result_csv["AD@10"] = (all_result_csv['ndcg@10'] * all_result_csv['novelty@10']) / \
            (all_result_csv['ndcg@10'] + all_result_csv['novelty@10'])
        all_result_csv["AD@15"] = (all_result_csv['ndcg@15'] * all_result_csv['novelty@15']) / \
            (all_result_csv['ndcg@15'] + all_result_csv['novelty@15'])
        all_result_csv.to_csv(self.config.csvPath)

        return res

    def BPRloss(self, positive, negative, mean=True, weight=None):
        loss = -torch.log(torch.sigmoid(positive-negative))
        if weight is not None:
            loss = loss * weight
            return loss.sum()
        else:
            if mean:
                return loss.mean()
            else:
                return loss.sum()

    def expandToSame(self, positive, negative):
        positive = positive.unsqueeze(1)
        negative = negative.unsqueeze(1)
        positive_new = positive.expand(-1, len(negative)).reshape((-1, 1))
        negative_new = negative.expand(-1, len(positive)).T.reshape((-1, 1))
        return positive_new, negative_new

    def linearRegression(self, X, Y):
        # X: (batch_size, seq_len, X_embedding_size)
        # Y: (batch_size, seq_len, Y_embedding_size)
        batch_size, seq_len, X_embedding_size = X.size()
        batch_size, seq_len, Y_embedding_size = Y.size()
        X = X.resize(batch_size * seq_len, X_embedding_size)
        Y = Y.resize(batch_size * seq_len, Y_embedding_size)
        X_T = X.T
        XTX = X_T.matmul(X)
        m = XTX.size()[0]
        I = torch.eye(m).to(self.device)
        solution = torch.inverse(XTX + I * self.config.ridge_lambd).matmul(
            X_T).matmul(Y)  # (X_embedding_size,Y_embedding_size)
        fit_value = torch.matmul(X, solution)
        residual = Y - fit_value

        fit_value = fit_value.resize(batch_size, seq_len, Y_embedding_size)
        residual = residual.resize(batch_size, seq_len, Y_embedding_size)
        return fit_value, residual

    def loss_fn(self, out_list, label, novelty, batch_behavior_id):
        if self.config.loss == "trade":
            ground_truth = self.config.trade_off * label + \
                (1 - self.config.trade_off) * novelty
            mseLossFn = nn.MSELoss()
            mseLoss = mseLossFn(
                out_list[0].squeeze(), ground_truth.float().squeeze())
            return mseLoss, torch.tensor(0., device=self.device)

        # if self.config.description == "SNPR":

        if self.config.weightedSum:
            click_score, novelty_score, interest_score = out_list
        elif self.config.naive:
            click_score, interest_score = out_list
        else:
            click_score = out_list[0]
        if self.config.loss == 'BCE':
            bceLossFn = nn.BCELoss()
            bceLoss = bceLossFn(click_score, label)
            if self.config.weightedSum:
                mseLossFn = nn.MSELoss()
                mseLoss = mseLossFn(
                    novelty_score, novelty) * self.config.interest_novelty_loss_weight
            else:
                mseLoss = torch.tensor(0., device=self.device)
            return bceLoss, mseLoss
        else:
            unique_behavior_id = batch_behavior_id.unique()
            click_loss = torch.tensor(0., device=self.device)
            novelty_loss = torch.tensor(0., device=self.device)
            click_count = 0
            novelty_count = 0
            for behavior_id in unique_behavior_id:
                curBehavior_mask = (batch_behavior_id == behavior_id)
                curBehavior_label = label[curBehavior_mask]
                positive_mask = (curBehavior_label == 1)
                negative_mask = (positive_mask == False)
                curBehavior_novelty = novelty[curBehavior_mask]
                curBehavior_click_score = click_score[curBehavior_mask]

                pair_positive_click_score, pair_negative_click_score = self.expandToSame(
                    curBehavior_click_score[positive_mask], curBehavior_click_score[negative_mask])
                click_loss += self.BPRloss(pair_positive_click_score,
                                           pair_negative_click_score, mean=False)
                click_count += len(pair_positive_click_score)
                del pair_positive_click_score, pair_negative_click_score

                # if self.config.naive or self.config.weightedSum:
                if self.config.loss == "listBPR":
                    curBehavior_interest_score = interest_score[curBehavior_mask]
                    pair_positive_novelty, pair_negative_novelty = self.expandToSame(
                        curBehavior_novelty[positive_mask], curBehavior_novelty[negative_mask])
                    first_novelty_mask = (
                        pair_positive_novelty > pair_negative_novelty)
                    second_novelty_mask = (
                        pair_positive_novelty < pair_negative_novelty)
                    del pair_positive_novelty, pair_negative_novelty

                    pair_positive_interest_score, pair_negative_interest_score = self.expandToSame(
                        curBehavior_interest_score[positive_mask], curBehavior_interest_score[negative_mask])
                    if second_novelty_mask.sum() > 0:
                        novelty_loss += self.BPRloss(pair_positive_interest_score[second_novelty_mask],
                                                     pair_negative_interest_score[second_novelty_mask], mean=False)
                        novelty_count += second_novelty_mask.sum()
                    del pair_positive_interest_score, pair_negative_interest_score

                # if self.config.weightedSum:
                if self.config.loss == "listBPR" and self.config.weightedSum:
                    curBehavior_novelty_score = novelty_score[curBehavior_mask]

                    pair_positive_novelty_score, pair_negative_novelty_score = self.expandToSame(
                        curBehavior_novelty_score[positive_mask], curBehavior_novelty_score[negative_mask])
                    if first_novelty_mask.sum() > 0:
                        novelty_loss += self.BPRloss(pair_positive_novelty_score[first_novelty_mask],
                                                     pair_negative_novelty_score[first_novelty_mask], mean=False)
                        novelty_count += first_novelty_mask.sum()
                    if second_novelty_mask.sum() > 0:
                        novelty_loss += self.BPRloss(pair_negative_novelty_score[second_novelty_mask],
                                                     pair_positive_novelty_score[second_novelty_mask], mean=False)
                        # novelty_loss -= self.BPRloss(pair_positive_novelty_score[second_novelty_mask],
                        #                              pair_negative_novelty_score[second_novelty_mask], mean=False)
                    del pair_positive_novelty_score, pair_negative_novelty_score

                # if self.config.loss == "listBPR":
                if self.config.loss == "listBPR" and self.config.weightedSum:
                    # rank novelty first
                    pair_positive_novelty_first, pair_positive_novelty_second = self.expandToSame(
                        curBehavior_novelty[positive_mask], curBehavior_novelty[positive_mask])
                    first_mask = (pair_positive_novelty_first >
                                  pair_positive_novelty_second)
                    del pair_positive_novelty_first, pair_positive_novelty_second

                    pair_positive_click_score_first, pair_positive_click_score_second = self.expandToSame(
                        curBehavior_click_score[positive_mask], curBehavior_click_score[positive_mask])
                    if first_mask.sum() > 0:
                        click_loss += self.BPRloss(pair_positive_click_score_first[first_mask],
                                                   pair_positive_click_score_second[first_mask], mean=False)*self.config.listBPRweight
                        click_count += first_mask.sum()
                    del pair_positive_click_score_first, pair_positive_click_score_second

                    if self.config.weightedSum:
                        pair_positive_novelty_score_first, pair_positive_novelty_score_second = self.expandToSame(
                            curBehavior_novelty_score[positive_mask], curBehavior_novelty_score[positive_mask])
                        if first_mask.sum() > 0:
                            novelty_loss += self.BPRloss(pair_positive_novelty_score_first[first_mask],
                                                         pair_positive_novelty_score_second[first_mask], mean=False)
                            novelty_count += first_mask.sum()
                        del pair_positive_novelty_score_first, pair_positive_novelty_score_second

            if click_count == 0:
                click_result = torch.tensor(0., device=self.device)
            else:
                click_result = click_loss / click_count
            if novelty_count == 0:
                novelty_result = torch.tensor(0., device=self.device)
            else:
                novelty_result = novelty_loss * \
                    self.config.interest_novelty_loss_weight / novelty_count
            return click_result, novelty_result
