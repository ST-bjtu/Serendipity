import torch
import torch.nn as nn
from config import GRU4Rec_config
from utils.layers import MLP

from models.basic_model import BasicModel


class GRU4Rec(BasicModel):
    def __init__(self, config: GRU4Rec_config):
        super().__init__(config)
        self.emb_size = config.itemEmbeddingSize
        self.hidden_size = config.hidden_size
        self.item_num = config.item_num
        self.prob_sigmoid = nn.Sigmoid()

        if self.config.preTrainEmbed:
            self.i_embeddings = nn.Embedding.from_pretrained(
                torch.load(config.itemEmbeddingPath), freeze=True)
            self.item_mlp = MLP(config.item_mlp)
        else:
            self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

        self.num_embedding = 2 if config.weightedSum else 1
        self.rnn_module = nn.ModuleList()
        for _ in range(self.num_embedding):
            temp_rnn = nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size,
                              num_layers=config.num_layers, batch_first=True)
            self.rnn_module.append(temp_rnn)
        # self.pred_embeddings = nn.Embedding(self.item_num, self.hidden_size)
        self.out_module = nn.ModuleList()
        for _ in range(self.num_embedding):
            temp_out = nn.Linear(self.hidden_size, self.emb_size)
            self.out_module.append(temp_out)

        self.initParameter()

    def initParameter(self):
        if self.config.dataset == "ml-1m":
            self.apply(self.init_weights)
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

    def getScore(self, lengths, his_vectors, pred_vectors, rnn_module, out_module):
        batch_size, seq_len, embed_size = his_vectors.size()

        # Sort and Pack
        sort_his_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
        sort_his_vectors = his_vectors.index_select(dim=0, index=sort_idx)
        history_packed = torch.nn.utils.rnn.pack_padded_sequence(
            sort_his_vectors, sort_his_lengths.cpu(), batch_first=True)
        # RNN
        output, hidden = rnn_module(history_packed, None)
        # Unsort
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
        rnn_vector = hidden[-1].index_select(dim=0, index=unsort_idx)
        # Predicts
        # pred_vectors = self.pred_embeddings(i_ids)
        rnn_vector = out_module(rnn_vector)
        prediction = (rnn_vector[:, None, :] * pred_vectors).sum(-1)
        prediction = self.prob_sigmoid(prediction.view(batch_size, -1))

        return prediction, rnn_vector

    def forward(self, user_history, target_item, novelty_value):
        user_history = user_history.long()
        target_item = target_item.long().unsqueeze(-1)

        i_ids = target_item  # [batch_size, -1]
        history = user_history  # [batch_size, history_max]
        lengths = (user_history != 0).sum(dim=1)  # [batch_size]
        lengths = lengths.to(self.device)

        his_vectors = self.getItemEmbedding(history)
        pred_vectors = self.getItemEmbedding(i_ids)

        prediction, _ = self.getScore(
            lengths, his_vectors, pred_vectors, self.rnn_module[0], self.out_module[0])
        return [prediction], torch.tensor(0., device=self.device)


class GRU4Rec_naiveNovelty(GRU4Rec):
    def __init__(self, config: GRU4Rec_config):
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

        his_vectors = self.getItemEmbedding(history)
        pred_vectors = self.getItemEmbedding(i_ids)

        interest_score, user_embedding = self.getScore(
            lengths, his_vectors, pred_vectors, self.rnn_module[0], self.out_module[0])
        novelty_score = novelty_value.unsqueeze(-1).to(self.device)

        weight_input = torch.cat([user_embedding, pred_vectors.squeeze()], dim=-1)
        novelty_alpha_weight = torch.softmax(
            self.alpha_novelty_interest(weight_input), dim=-1)

        novelty_weight = novelty_alpha_weight[:, 0].unsqueeze(-1)
        interest_weight = novelty_alpha_weight[:, 1].unsqueeze(-1)

        # weight_input = user_embedding

        # novelty_weight = self.prob_sigmoid(self.alpha_novelty(weight_input))
        # interest_weight = self.prob_sigmoid(self.alpha_interest(weight_input))

        click_score = novelty_weight*novelty_score+interest_weight*interest_score
        return [click_score, interest_score], torch.tensor(0., device=self.device)


class GRU4Rec_DICE(GRU4Rec):
    def __init__(self, config: GRU4Rec_config):
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

        his_vectors_novelty, his_vectors_interest, his_vectors_discrepency = self.getItemEmbedding(
            history)
        pred_vectors_novelty, pred_vectors_interest, pred_vectors_discrepency = self.getItemEmbedding(
            i_ids)

        novelty_score, user_novelty = self.getScore(
            lengths, his_vectors_novelty, pred_vectors_novelty, self.rnn_module[0], self.out_module[0])
        interest_score, user_interest = self.getScore(
            lengths, his_vectors_interest, pred_vectors_interest, self.rnn_module[0], self.out_module[0])

        weight_input = torch.cat(
            [user_novelty, pred_vectors_novelty.squeeze(), user_interest, pred_vectors_interest.squeeze()], dim=-1)
        novelty_alpha_weight = torch.softmax(
            self.alpha_novelty_interest(weight_input), dim=-1)

        novelty_weight = novelty_alpha_weight[:, 0].unsqueeze(-1)
        interest_weight = novelty_alpha_weight[:, 1].unsqueeze(-1)

        # weight_input = torch.cat([user_novelty, user_interest], dim=-1)

        # novelty_weight = self.prob_sigmoid(self.alpha_novelty(weight_input))
        # interest_weight = self.prob_sigmoid(self.alpha_interest(weight_input))

        click_score = novelty_weight*novelty_score+interest_weight*interest_score

        return [click_score, novelty_score, interest_score], torch.tensor(0., device=self.device)


class GRU4Rec_IVmediate(GRU4Rec_DICE):
    def __init__(self, config: GRU4Rec_config):
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


class GRU4Rec_IVconcat(GRU4Rec_IVmediate):
    def __init__(self, config: GRU4Rec_config):
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


class GRU4Rec_IVmediate_re(GRU4Rec_IVmediate):
    def __init__(self, config: GRU4Rec_config):
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


class GRU4Rec_IVmediate_re_true(GRU4Rec_IVmediate):
    def __init__(self, config: GRU4Rec_config):
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
