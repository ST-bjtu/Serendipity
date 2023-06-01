import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.linalg
import torch.nn as nn
from tqdm import tqdm
from utils.layers import *
from utils.metric import cal_metric

from models.basic_model import BasicModel


class IVmediate_linear(BasicModel):
    def __init__(self, config):
        super().__init__(config)

        # self.itemIVembedding = nn.Embedding.from_pretrained(
        #     torch.load(config.itemIVpath), freeze=True)
        self.itemEmbedding = nn.Embedding.from_pretrained(
            torch.load(config.itemEmbeddingPath), freeze=True)
        # self.itemIV_mlp = MLP(config.itemIV_mlp)
        self.item_mlp = MLP(config.item_mlp)

        # self.noveltyNet = MLP(config.noveltyMLP)
        self.alpha_interest = MLP(config.alpha_interest)
        self.alpha_novelty = MLP(config.alpha_novelty)
        self.prob_sigmoid = nn.Sigmoid()

        self.cos = nn.CosineSimilarity()
        # self.click_linear = nn.Linear(
        #     in_features=2, out_features=1, bias=False)

        for m in self.children():
            # embedding太大 GPU会out of memory
            if isinstance(m, nn.Embedding):
                continue
            m.to(self.device)

    def forward(self, user_history, target_item, novelty_score, print_weight=False):
        novelty_score = novelty_score.unsqueeze(1).to(self.device)

        user_history_mask = torch.where(
            user_history == 0, 1, 0).bool().to(self.device)
        user_history_embedding = self.itemEmbedding(
            user_history.long()).to(self.device)
        user_history_embedding = self.item_mlp(user_history_embedding)
        batch_size, seq_len, itemEmbeddingSize = user_history_embedding.size()

        user_history_emb_mask = user_history_mask.unsqueeze(
            2).expand(batch_size, seq_len, itemEmbeddingSize)

        user_history_embedding = user_history_embedding.masked_fill(
            user_history_emb_mask, 0)

        target_item_embedding = self.itemEmbedding(
            target_item.long().unsqueeze(1)).to(self.device)
        target_item_embedding = self.item_mlp(target_item_embedding)

        user_embedding = user_history_embedding.sum(
            dim=1) / (user_history_emb_mask == False).sum(dim=1)

        interest_score = self.cos(
            user_embedding, target_item_embedding.squeeze()).unsqueeze(1)

        novelty_weight = self.prob_sigmoid(self.alpha_novelty(user_embedding))
        interest_weight = self.prob_sigmoid(
            self.alpha_interest(user_embedding))

        if print_weight:
            return novelty_weight, interest_weight

        click_score = novelty_weight*novelty_score+interest_weight*interest_score
        return click_score

    def loss_fn(self, batch_data):
        if self.config.loss == 'BCE':
            loss = nn.BCELoss()
            return loss(batch_data[:, 0], batch_data[:, 1].detach())
        elif self.config.loss == "BPR":
            unique_behavior_id = batch_data[:, -1].unique()
            click_loss = torch.tensor(0., device=self.device)
            click_count = 0
            for behavior_id in unique_behavior_id:
                curBehavior_data = batch_data[batch_data[:, -1] == behavior_id]
                positive_mask = (curBehavior_data[:, -3] == 1)
                negative_mask = (positive_mask == False)
                positive_data = curBehavior_data[positive_mask]
                negative_data = curBehavior_data[negative_mask]

                pair_positive_click_score, pair_negative_click_score = self.expandToSame(
                    positive_data[:, 0].unsqueeze(1), negative_data[:, 0].unsqueeze(1))
                click_loss += self.BPRloss(pair_positive_click_score,
                                           pair_negative_click_score, mean=False)
                click_count += len(pair_positive_click_score)
            if click_count == 0:
                click_result = torch.tensor(0., device=self.device)
            else:
                click_result = click_loss / click_count
            return click_result
        elif self.config.loss == 'listBPR':
            unique_behavior_id = batch_data[:, -1].unique()
            click_loss = torch.tensor(0., device=self.device)
            click_count = 0
            for behavior_id in unique_behavior_id:
                curBehavior_data = batch_data[batch_data[:, -1] == behavior_id]
                positive_mask = (curBehavior_data[:, -3] == 1)
                negative_mask = (positive_mask == False)
                positive_data = curBehavior_data[positive_mask]
                negative_data = curBehavior_data[negative_mask]

                pair_positive_click_score, pair_negative_click_score = self.expandToSame(
                    positive_data[:, 0].unsqueeze(1), negative_data[:, 0].unsqueeze(1))
                click_loss += self.BPRloss(pair_positive_click_score,
                                           pair_negative_click_score, mean=False)
                click_count += len(pair_positive_click_score)

                # rank novelty first
                pair_positive_novelty_first, pair_positive_novelty_second = self.expandToSame(
                    positive_data[:, -2].unsqueeze(1), positive_data[:, -2].unsqueeze(1))
                pair_positive_click_score_first, pair_positive_click_score_second = self.expandToSame(
                    positive_data[:, 0].unsqueeze(1), positive_data[:, 0].unsqueeze(1))
                first_mask = (pair_positive_novelty_first >
                              pair_positive_novelty_second)
                if first_mask.sum() > 0:
                    click_loss += self.BPRloss(pair_positive_click_score_first[first_mask],
                                               pair_positive_click_score_second[first_mask], mean=False)
                    click_count += first_mask.sum()
            if click_count == 0:
                click_result = torch.tensor(0., device=self.device)
            else:
                click_result = click_loss / click_count
            return click_result
        else:
            raise ValueError("loss error!")

    def _run_train(self, dataloader, optimizer, cur_epoch):

        print("==========epoch:{}==========".format(cur_epoch))
        epoch_loss = 0
        tqdm_ = tqdm(iterable=dataloader)

        for step, batch in enumerate(tqdm_):
            history, user, target_item, label, novelty, popularity, behavior_id = batch
            output = self(history, target_item, novelty)
            curResult = torch.cat([output, label.unsqueeze(1).to(self.device), novelty.unsqueeze(1).to(
                self.device), behavior_id.unsqueeze(1).to(self.device)], dim=-1)
            click_loss = self.loss_fn(curResult)
            loss = click_loss

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % self.config.print_interval == 0 and step > 0:
                tqdm_.set_description(
                    "epoch:{:d} step:{:d} loss:{:.4f}".format(
                        cur_epoch, step, epoch_loss / step
                    ))
        modelInfo = self.config.modelInfo + \
            " " + "epoch_{}.pth".format(cur_epoch)
        torch.save(self.state_dict(), os.path.join(
            self.config.finalPath, modelInfo))

    def _run_eval(self, dataloader):
        preds = []
        labels = []
        novelty_list = []
        for batch in tqdm(dataloader):
            history, user, target_item, label, novelty, popularity, behavior_id = batch
            output = self(history, target_item, novelty)
            pred = output

            unique_behavior_id = behavior_id.unique()
            for cur_behavior_id in unique_behavior_id:
                cur_mask = behavior_id == cur_behavior_id
                cur_behavior_pred = pred[cur_mask]
                cur_behavior_label = label[cur_mask]
                cur_behavior_novelty = novelty[cur_mask]

                preds.append(cur_behavior_pred.squeeze().tolist())
                labels.append(cur_behavior_label.squeeze().tolist())
                novelty_list.append(cur_behavior_novelty.squeeze().tolist())

        return preds, labels, novelty_list

    def printWeight(self, dataloader, load_epoch):
        modelInfo = self.config.modelInfo + " epoch_{}.pth".format(load_epoch)
        print("==========load_model_from :{}==========".format(modelInfo))
        self.load_state_dict(torch.load(os.path.join(
            self.config.finalPath, modelInfo), map_location=self.device))

        tqdm_ = tqdm(iterable=dataloader)
        all_user_novelty = None
        all_user_interest = None
        for step, batch in enumerate(tqdm_):
            history, user, target_item, label, novelty, popularity, behavior_id = batch
            novelty_weight, interest_weight = self(
                history, target_item, novelty, print_weight=True)

            novelty_weight = novelty_weight.detach().cpu().numpy()
            interest_weight = interest_weight.detach().cpu().numpy()

            all_user_novelty = novelty_weight if all_user_novelty is None else np.concatenate(
                (all_user_novelty, novelty_weight), axis=0)
            all_user_interest = interest_weight if all_user_interest is None else np.concatenate(
                (all_user_interest, interest_weight), axis=0)
        np.save("Figure/IVmediate_linear_novelty.npy", all_user_novelty)
        np.save("Figure/IVmediate_linear_interest.npy", all_user_interest)
