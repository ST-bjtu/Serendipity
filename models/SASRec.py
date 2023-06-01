import torch
import torch.nn as nn
import numpy as np
from config import SASRec_config
from utils.layers import TransformerLayer, MLP

from models.basic_model import BasicModel


class SASRec(BasicModel):
    def __init__(self, config: SASRec_config):
        super().__init__(config)
        self.item_num = config.item_num
        self.emb_size = config.itemEmbeddingSize
        self.max_his = config.pad_len
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.len_range = torch.from_numpy(
            np.arange(self.max_his)).to(self.device)
        self.prob_sigmoid = nn.Sigmoid()

        if self.config.preTrainEmbed:
            self.i_embeddings = nn.Embedding.from_pretrained(
                torch.load(config.itemEmbeddingPath), freeze=True)
            self.item_mlp = MLP(self.config.item_mlp)
        else:
            self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)

        self.num_embedding = 2 if config.weightedSum else 1
        self.transformer_block_module = nn.ModuleList()
        for _ in range(self.num_embedding):
            temp_transformer_block = nn.ModuleList([
                TransformerLayer(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.num_heads,
                                 dropout=self.dropout, kq_same=False)
                for _ in range(self.num_layers)
            ])
            self.transformer_block_module.append(temp_transformer_block)

        self.initParameter()

    def initParameter(self):
        if self.config.dataset == "ml-1m":
            self.apply(self.init_weights)
        # elif self.config.dataset == "MINDsmall":
        #     for m in self.children():
        #         if isinstance(m, nn.Linear):
        #             nn.init.xavier_normal_(m.weight)

        for m in self.children():
            # embedding太大 GPU会out of memory
            if isinstance(m, nn.Embedding):
                continue
            m.to(self.device)

    @staticmethod
    def init_weights(m):
        if 'Linear' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def getItemEmbedding(self, treatment):
        treatmentEmbedding = self.i_embeddings(treatment.long()).to(
            self.device)  # (batch_size, seq_len,itemEmbeddingSize)
        if self.config.preTrainEmbed:
            treatmentEmbedding = self.item_mlp(treatmentEmbedding)
        return treatmentEmbedding

    def getScore(self, lengths, valid_his, his_vectors, transformer_block, i_vectors):
        batch_size, seq_len, embed_size = his_vectors.size()

        # Self-attention
        causality_mask = np.tril(
            np.ones((1, 1, seq_len, seq_len), dtype=np.int))
        attn_mask = torch.from_numpy(causality_mask).to(self.device)
        # attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
        for block in transformer_block:
            his_vectors = block(his_vectors, attn_mask)
        his_vectors = his_vectors * valid_his[:, :, None].float()

        his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :]
        # his_vector (batch_size, embed_size)
        # his_vector = his_vectors.sum(1) / lengths[:, None].float()
        # ↑ average pooling is shown to be more effective than the most recent embedding

        # his_vector[:, None, :] (batch_size, 1, embed_size)
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        prediction = self.prob_sigmoid(prediction.view(batch_size, -1))

        return prediction, his_vector

    def forward(self, user_history, target_item, novelty_value):
        user_history = user_history.long()
        target_item = target_item.long().unsqueeze(-1)

        i_ids = target_item  # [batch_size, -1]
        history = user_history  # [batch_size, history_max]
        lengths = (user_history != 0).sum(dim=1)  # [batch_size]
        lengths = lengths.to(self.device)
        batch_size, seq_len = history.shape

        valid_his = (history > 0).long().to(self.device)
        his_vectors = self.getItemEmbedding(history)

        # Position embedding
        # lengths:  [4, 2, 5]
        # position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
        position = (lengths[:, None] -
                    self.len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position.cpu()).to(self.device)
        his_vectors = his_vectors + pos_vectors

        i_vectors = self.getItemEmbedding(i_ids)

        prediction, _ = self.getScore(
            lengths, valid_his, his_vectors, self.transformer_block_module[0], i_vectors)

        return [prediction], torch.tensor(0., device=self.device)


class SASRec_naiveNovelty(SASRec):
    def __init__(self, config: SASRec_config):
        super().__init__(config)
        self.alpha_novelty_interest = MLP(config.alpha_novelty_interest)
        # self.alpha_interest = MLP(config.alpha_interest)
        # self.alpha_novelty = MLP(config.alpha_novelty)
        self.prob_sigmoid = nn.Sigmoid()

        self.initParameter()

    def forward(self, user_history, target_item, novelty_value):
        user_history = user_history.long()
        target_item = target_item.long().unsqueeze(-1)

        i_ids = target_item  # [batch_size, -1]
        history = user_history  # [batch_size, history_max]
        lengths = (user_history != 0).sum(dim=1)  # [batch_size]
        lengths = lengths.to(self.device)
        batch_size, seq_len = history.shape

        valid_his = (history > 0).long().to(self.device)
        his_vectors = self.getItemEmbedding(history)

        # Position embedding
        # lengths:  [4, 2, 5]
        # position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
        position = (lengths[:, None] -
                    self.len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position.cpu()).to(self.device)
        his_vectors = his_vectors + pos_vectors

        i_vectors = self.getItemEmbedding(i_ids)

        interest_score, user_embedding = self.getScore(
            lengths, valid_his, his_vectors, self.transformer_block_module[0], i_vectors)
        novelty_score = novelty_value.unsqueeze(-1).to(self.device)

        weight_input = torch.cat([user_embedding, i_vectors.squeeze()], dim=-1)
        novelty_alpha_weight = torch.softmax(
            self.alpha_novelty_interest(weight_input), dim=-1)

        novelty_weight = novelty_alpha_weight[:, 0].unsqueeze(-1)
        interest_weight = novelty_alpha_weight[:, 1].unsqueeze(-1)

        # weight_input = user_embedding

        # novelty_weight = self.prob_sigmoid(self.alpha_novelty(weight_input))
        # interest_weight = self.prob_sigmoid(self.alpha_interest(weight_input))

        click_score = novelty_weight*novelty_score+interest_weight*interest_score

        return [click_score, interest_score], torch.tensor(0., device=self.device)


class SASRec_DICE(SASRec):
    def __init__(self, config: SASRec_config):
        super().__init__(config)
        self.noveltyNet = MLP(config.noveltyMLP)
        self.alpha_novelty_interest = MLP(config.alpha_novelty_interest)
        # self.alpha_interest = MLP(config.alpha_interest)
        # self.alpha_novelty = MLP(config.alpha_novelty)

        self.initParameter()

    def getItemEmbedding(self, treatment):
        treatmentEmbedding = self.i_embeddings(treatment.long()).to(
            self.device)  # (batch_size, seq_len,itemEmbeddingSize)
        if self.config.preTrainEmbed:
            treatmentEmbedding = self.item_mlp(treatmentEmbedding)
        interest_embedding = treatmentEmbedding
        novelty_embedding = self.noveltyNet(treatmentEmbedding)
        discrepency_loss = torch.tensor(0., device=self.device)

        return novelty_embedding, interest_embedding, -discrepency_loss

    def forward(self, user_history, target_item, novelty_value):
        user_history = user_history.long()
        target_item = target_item.long().unsqueeze(-1)

        i_ids = target_item  # [batch_size, -1]
        history = user_history  # [batch_size, history_max]
        lengths = (user_history != 0).sum(dim=1)  # [batch_size]
        lengths = lengths.to(self.device)
        batch_size, seq_len = history.shape

        valid_his = (history > 0).long().to(self.device)
        his_vectors_novelty, his_vectors_interest, his_vectors_discrepency = self.getItemEmbedding(
            history)

        # Position embedding
        # lengths:  [4, 2, 5]
        # position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
        position = (lengths[:, None] -
                    self.len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position.cpu()).to(self.device)

        his_vectors_novelty = his_vectors_novelty + pos_vectors
        his_vectors_interest = his_vectors_interest + pos_vectors

        i_vectors_novelty, i_vectors_interest, i_vectors_discrepency = self.getItemEmbedding(
            i_ids)

        novelty_score, user_novelty = self.getScore(
            lengths, valid_his, his_vectors_novelty, self.transformer_block_module[0], i_vectors_novelty)

        interest_score, user_interest = self.getScore(
            lengths, valid_his, his_vectors_interest, self.transformer_block_module[1], i_vectors_interest)

        weight_input = torch.cat(
            [user_novelty, i_vectors_novelty.squeeze(), user_interest, i_vectors_interest.squeeze()], dim=-1)
        novelty_alpha_weight = torch.softmax(
            self.alpha_novelty_interest(weight_input), dim=-1)

        novelty_weight = novelty_alpha_weight[:, 0].unsqueeze(-1)
        interest_weight = novelty_alpha_weight[:, 1].unsqueeze(-1)

        # weight_input = torch.cat([user_novelty, user_interest], dim=-1)

        # novelty_weight = self.prob_sigmoid(self.alpha_novelty(weight_input))
        # interest_weight = self.prob_sigmoid(self.alpha_interest(weight_input))

        click_score = novelty_weight*novelty_score+interest_weight*interest_score

        return [click_score, novelty_score, interest_score], torch.tensor(0., device=self.device)


class SASRec_IVmediate(SASRec_DICE):
    def __init__(self, config: SASRec_config):
        super().__init__(config)
        self.itemIVembedding = nn.Embedding.from_pretrained(
            torch.load(config.itemIVEmbeddingPath), freeze=True)
        self.itemIV_mlp = MLP(config.itemIV_mlp)

        self.initParameter()

    def getItemEmbedding(self, treatment):
        # treatment: (batch_size, seq_len)
        ivEmbedding = self.itemIVembedding(treatment.long()).to(
            self.device)  # (batch_size, seq_len,ivEmbeddingSize)
        # ivEmbedding = self.IVMlp(ivEmbedding)
        treatmentEmbedding = self.i_embeddings(treatment.long()).to(
            self.device)  # (batch_size, seq_len,itemEmbeddingSize)
        ivEmbedding = self.itemIV_mlp(ivEmbedding)
        if self.config.preTrainEmbed:
            treatmentEmbedding = self.item_mlp(treatmentEmbedding)

        novelty_embedding = self.noveltyNet(treatmentEmbedding)
        if self.config.add_residual:
            return novelty_embedding, treatmentEmbedding
        else:
            novelty_IV = torch.cat([ivEmbedding, treatmentEmbedding], dim=-1)
            novelty_embedding_fit, novelty_embedding_residual = self.linearRegression(
                novelty_IV, novelty_embedding)

            return novelty_embedding_fit, treatmentEmbedding, torch.tensor(0., device=self.device)


class SASRec_IVconcat(SASRec_IVmediate):
    def __init__(self, config: SASRec_config):
        super().__init__(config)
        self.ivConcat_mlp = MLP(config.ivConcat_mlp)

        self.initParameter()

    def getItemEmbedding(self, treatment):
        # treatment: (batch_size, seq_len)
        ivEmbedding = self.itemIVembedding(treatment.long()).to(
            self.device)  # (batch_size, seq_len,ivEmbeddingSize)
        ivEmbedding = self.itemIV_mlp(ivEmbedding)
        treatmentEmbedding = self.i_embeddings(treatment.long()).to(
            self.device)  # (batch_size, seq_len,itemEmbeddingSize)
        if self.config.preTrainEmbed:
            treatmentEmbedding = self.item_mlp(treatmentEmbedding)

        newTreatmentEmbedding = torch.cat(
            [ivEmbedding, treatmentEmbedding], dim=-1)
        newTreatmentEmbedding = self.ivConcat_mlp(newTreatmentEmbedding)
        novelty_embedding = self.noveltyNet(newTreatmentEmbedding)

        return novelty_embedding, newTreatmentEmbedding, torch.tensor(0., device=self.device)


class SASRec_IVmediate_re(SASRec_IVmediate):
    def __init__(self, config: SASRec_config):
        super().__init__(config)
        self.novelty_alpha_1 = MLP(config.alpha_1)
        self.novelty_alpha_2 = MLP(config.alpha_2)

        self.initParameter()

    def getItemEmbedding(self, treatment):
        # treatment: (batch_size, seq_len)
        ivEmbedding = self.itemIVembedding(treatment.long()).to(
            self.device)  # (batch_size, seq_len,ivEmbeddingSize)
        treatmentEmbedding = self.i_embeddings(treatment.long()).to(
            self.device)  # (batch_size, seq_len,itemEmbeddingSize)
        ivEmbedding = self.itemIV_mlp(ivEmbedding)
        if self.config.preTrainEmbed:
            treatmentEmbedding = self.item_mlp(treatmentEmbedding)

        novelty_embedding = self.noveltyNet(treatmentEmbedding)

        novelty_IV = torch.cat([ivEmbedding, treatmentEmbedding], dim=-1)
        novelty_embedding_fit, novelty_embedding_residual = self.linearRegression(
            novelty_IV, novelty_embedding)
        alpha_input = torch.cat(
            [novelty_embedding_fit, novelty_embedding_residual], dim=-1)
        novelty_alpha_1 = self.novelty_alpha_1(alpha_input)
        novelty_alpha_2 = self.novelty_alpha_2(alpha_input)
        novelty_embedding = novelty_alpha_1 * novelty_embedding_fit + \
            novelty_alpha_2 * novelty_embedding_residual

        return novelty_embedding, treatmentEmbedding, torch.tensor(0., device=self.device)


class SASRec_IVmediate_re_true(SASRec_IVmediate):
    def __init__(self, config: SASRec_config):
        super().__init__(config)
        self.novelty_alpha_1 = MLP(config.alpha_1)
        self.novelty_alpha_2 = MLP(config.alpha_2)

        self.initParameter()

    def getItemEmbedding(self, treatment):
        # treatment: (batch_size, seq_len)
        ivEmbedding = self.itemIVembedding(treatment.long()).to(
            self.device)  # (batch_size, seq_len,ivEmbeddingSize)
        treatmentEmbedding = self.i_embeddings(treatment.long()).to(
            self.device)  # (batch_size, seq_len,itemEmbeddingSize)
        ivEmbedding = self.itemIV_mlp(ivEmbedding)
        if self.config.preTrainEmbed:
            treatmentEmbedding = self.item_mlp(treatmentEmbedding)

        novelty_embedding = self.noveltyNet(treatmentEmbedding)

        novelty_IV = torch.cat([ivEmbedding, treatmentEmbedding], dim=-1)
        novelty_embedding_fit, novelty_embedding_residual = self.linearRegression(
            novelty_IV, novelty_embedding)
        alpha_input = torch.cat(
            [novelty_embedding, novelty_IV], dim=-1)
        novelty_alpha_1 = self.novelty_alpha_1(alpha_input)
        novelty_alpha_2 = self.novelty_alpha_2(alpha_input)
        novelty_embedding = novelty_alpha_1 * novelty_embedding_fit + \
            novelty_alpha_2 * novelty_embedding_residual

        return novelty_embedding, treatmentEmbedding, torch.tensor(0., device=self.device)