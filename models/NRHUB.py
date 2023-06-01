import torch
import torch.nn as nn
from utils.layers import *
from config import NRHUB_Config
from models.basic_model import BasicModel


class NRHUB(BasicModel):
    def __init__(self, config: NRHUB_Config):
        super().__init__(config)
        if config.preTrainEmbed:
            self.itemEmbedding = nn.Embedding.from_pretrained(
                torch.load(config.itemEmbeddingPath), freeze=True)
            self.item_mlp = MLP(config.item_mlp)
        else:
            self.itemEmbedding = nn.Embedding(
                config.item_num, config.itemEmbeddingSize)

        self.num_embedding = 2 if config.weightedSum else 1

        self.user_history_attention = nn.ModuleList()
        for i in range(self.num_embedding):
            temp_attention = Attention(
                config.pad_len, config.itemEmbeddingSize, config.Dense_dim)
            self.user_history_attention.append(temp_attention)
        self.item_rep = nn.ModuleList()
        for i in range(self.num_embedding):
            temp_item_rep = nn.Sequential(
                nn.Linear(config.itemEmbeddingSize, config.Dense_dim),
                nn.Tanh()
            )
            self.item_rep.append(temp_item_rep)
        self.user_rep = nn.ModuleList()
        for i in range(self.num_embedding):
            temp_user_rep = nn.Sequential(
                nn.Linear(config.itemEmbeddingSize, config.Dense_dim),
                nn.Tanh(),
                Attention(1, config.Dense_dim, 100),
            )
            self.user_rep.append(temp_user_rep)

        self.prob_sigmoid = nn.Sigmoid()
        self.initParameter()

    def initParameter(self):
        if self.config.dataset == "MINDsmall":
            for m in self.children():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        elif self.config.dataset == "ml-1m":
            self.apply(self.init_weights)
        for m in self.children():
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
        treatmentEmbedding = self.itemEmbedding(treatment.long()).to(
            self.device)  # (batch_size, seq_len,itemEmbeddingSize)
        if self.config.preTrainEmbed:
            treatmentEmbedding = self.item_mlp(treatmentEmbedding)
        return treatmentEmbedding

    def forward(self, user_history, target_item, novelty_value):
        # user_history: (batch_size, seq_len)
        user_history_mask = torch.where(
            user_history == 0, 1, 0).bool().to(self.device)
        user_history_embedding = self.getItemEmbedding(user_history)

        target_item_embedding = self.getItemEmbedding(target_item.unsqueeze(1))
        target_item_rep = self.item_rep[0](
            target_item_embedding)  # (batch_size, 1, Dense_dim)

        user_history_rep = self.user_history_attention[0](
            user_history_embedding, user_history_mask)  # (batch_size, itemEmbeddingSize)

        # (batch_size, 1, itemEmbeddingSize)
        # (batch_size, Dense_dim, 1)
        user_rep = self.user_rep[0](user_history_rep.unsqueeze_(dim=1))
        prob = self.prob_sigmoid(torch.matmul(
            target_item_rep, user_rep.unsqueeze(dim=-1))).squeeze(-1)

        # (batch_size, 1)
        return [prob], torch.tensor(0., device=self.device)


class NRHUB_naiveNovelty(NRHUB):
    def __init__(self, config: NRHUB_Config):
        super().__init__(config)
        self.alpha_novelty_interest = MLP(config.alpha_novelty_interest)
        # self.alpha_interest = MLP(config.alpha_interest)
        # self.alpha_novelty = MLP(config.alpha_novelty)
        self.prob_sigmoid = nn.Sigmoid()

        self.initParameter()

    def forward(self, user_history, target_item, novelty_value):
        # user_history: (batch_size, seq_len)
        user_history_mask = torch.where(
            user_history == 0, 1, 0).bool().to(self.device)
        user_history_embedding = self.getItemEmbedding(user_history)

        target_item_embedding = self.getItemEmbedding(target_item.unsqueeze(1))
        target_item_rep = self.item_rep[0](
            target_item_embedding)  # (batch_size, 1, Dense_dim)

        user_history_rep = self.user_history_attention[0](
            user_history_embedding, user_history_mask)  # (batch_size, itemEmbeddingSize)

        # (batch_size, 1, itemEmbeddingSize)
        # (batch_size, Dense_dim, 1)
        user_rep = self.user_rep[0](user_history_rep.unsqueeze_(dim=1))

        interest_score = self.prob_sigmoid(torch.matmul(
            target_item_rep, user_rep.unsqueeze(dim=-1))).squeeze(-1)
        novelty_score = novelty_value.unsqueeze(-1).to(self.device)

        weight_input = torch.cat([user_rep, target_item_rep.squeeze()], dim=-1)
        novelty_alpha_weight = torch.softmax(
            self.alpha_novelty_interest(weight_input), dim=-1)
        novelty_weight = novelty_alpha_weight[:, 0].unsqueeze(-1)
        interest_weight = novelty_alpha_weight[:, 1].unsqueeze(-1)

        # weight_input = user_rep
        # novelty_weight = self.prob_sigmoid(self.alpha_novelty(weight_input))
        # interest_weight = self.prob_sigmoid(self.alpha_interest(weight_input))
        click_score = novelty_weight * novelty_score + interest_weight * interest_score

        return [click_score, interest_score], torch.tensor(0., device=self.device)


class NRHUB_DICE(NRHUB):
    def __init__(self, config: NRHUB_Config):
        super().__init__(config)
        self.noveltyNet = MLP(config.noveltyMLP)
        self.alpha_novelty_interest = MLP(config.alpha_novelty_interest)
        # self.alpha_interest = MLP(config.alpha_interest)
        # self.alpha_novelty = MLP(config.alpha_novelty)
        self.prob_sigmoid = nn.Sigmoid()

        if config.dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif config.dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()
        elif config.dis_loss == 'dcor':
            self.criterion_discrepancy = self.dcor
        else:
            raise ValueError("dis_loss not exist!")

        self.initParameter()

    def getItemEmbedding(self, treatment):
        treatmentEmbedding = self.itemEmbedding(treatment.long()).to(
            self.device)  # (batch_size, seq_len,itemEmbeddingSize)
        if self.config.preTrainEmbed:
            treatmentEmbedding = self.item_mlp(treatmentEmbedding)
        interest_embedding = treatmentEmbedding
        novelty_embedding = self.noveltyNet(treatmentEmbedding)
        discrepency_loss = self.criterion_discrepancy(
            interest_embedding, novelty_embedding)

        return novelty_embedding, interest_embedding, -discrepency_loss

    def forward(self, user_history, target_item, novelty_value):
        # user_history: (batch_size, seq_len)
        user_history_mask = torch.where(
            user_history == 0, 1, 0).bool().to(self.device)
        user_history_novelty, user_history_interest, history_discrepency = self.getItemEmbedding(
            user_history)  # (batch_size, seq_len, itemEmbeddingSize)

        target_item_novelty, target_item_interest, target_item_discrepency = self.getItemEmbedding(
            target_item.unsqueeze(1))  # (batch_size, 1,itemEmbeddingSize)
        target_item_novelty_rep = self.item_rep[0](
            target_item_novelty)  # (batch_size, 1, Dense_dim)
        target_item_interest_rep = self.item_rep[1](target_item_interest)

        user_history_novelty_rep = self.user_history_attention[0](
            user_history_novelty, user_history_mask)
        user_novelty_rep = self.user_rep[0](
            user_history_novelty_rep.unsqueeze_(dim=1))  # (batch_size, Dense_dim)
        novelty_score = self.prob_sigmoid(torch.matmul(
            target_item_novelty_rep, user_novelty_rep.unsqueeze(dim=-1))).squeeze(-1)

        user_history_interest_rep = self.user_history_attention[1](
            user_history_interest, user_history_mask)
        user_interest_rep = self.user_rep[1](
            user_history_interest_rep.unsqueeze_(dim=1))  # (batch_size, Dense_dim)
        interest_score = self.prob_sigmoid(torch.matmul(
            target_item_interest_rep, user_interest_rep.unsqueeze(dim=-1))).squeeze(-1)

        weight_input = torch.cat([user_novelty_rep, target_item_novelty_rep.squeeze(
        ), user_interest_rep, target_item_interest_rep.squeeze()], dim=-1)
        novelty_alpha_weight = torch.softmax(
            self.alpha_novelty_interest(weight_input), dim=-1)
        novelty_weight = novelty_alpha_weight[:, 0].unsqueeze(-1)
        interest_weight = novelty_alpha_weight[:, 1].unsqueeze(-1)

        # weight_input = torch.cat([user_novelty_rep, user_interest_rep], dim=-1)
        # novelty_weight = self.prob_sigmoid(self.alpha_novelty(weight_input))
        # interest_weight = self.prob_sigmoid(self.alpha_interest(weight_input))

        click_score = novelty_weight*novelty_score+interest_weight*interest_score
        discrepency_loss = (history_discrepency + target_item_discrepency) / 2

        return [click_score, novelty_score, interest_score], discrepency_loss


class NRHUB_IVmediate(NRHUB_DICE):
    def __init__(self, config: NRHUB_Config):
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
        treatmentEmbedding = self.itemEmbedding(treatment.long()).to(
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


class NRHUB_IVconcat(NRHUB_IVmediate):
    def __init__(self, config: NRHUB_Config):
        super().__init__(config)
        self.ivConcat_mlp = MLP(config.ivConcat_mlp)
        self.initParameter()

    def getItemEmbedding(self, treatment):
        # treatment: (batch_size, seq_len)
        ivEmbedding = self.itemIVembedding(treatment.long()).to(
            self.device)  # (batch_size, seq_len,ivEmbeddingSize)
        ivEmbedding = self.itemIV_mlp(ivEmbedding)
        treatmentEmbedding = self.itemEmbedding(treatment.long()).to(
            self.device)  # (batch_size, seq_len,itemEmbeddingSize)
        if self.config.preTrainEmbed:
            treatmentEmbedding = self.item_mlp(treatmentEmbedding)

        newTreatmentEmbedding = torch.cat(
            [ivEmbedding, treatmentEmbedding], dim=-1)
        newTreatmentEmbedding = self.ivConcat_mlp(newTreatmentEmbedding)
        novelty_embedding = self.noveltyNet(newTreatmentEmbedding)

        return novelty_embedding, newTreatmentEmbedding, torch.tensor(0., device=self.device)


class NRHUB_IVmediate_re(NRHUB_IVmediate):
    def __init__(self, config: NRHUB_Config):
        super().__init__(config)
        self.novelty_alpha_1 = MLP(config.alpha_1)
        self.novelty_alpha_2 = MLP(config.alpha_2)

        self.initParameter()

    def getItemEmbedding(self, treatment):
        # treatment: (batch_size, seq_len)
        ivEmbedding = self.itemIVembedding(treatment.long()).to(
            self.device)  # (batch_size, seq_len,ivEmbeddingSize)
        # ivEmbedding = self.IVMlp(ivEmbedding)
        treatmentEmbedding = self.itemEmbedding(treatment.long()).to(
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
            alpha_input = torch.cat(
                [novelty_embedding_fit, novelty_embedding_residual], dim=-1)
            novelty_alpha_1 = self.novelty_alpha_1(alpha_input)
            novelty_alpha_2 = self.novelty_alpha_2(alpha_input)
            novelty_embedding = novelty_alpha_1 * novelty_embedding_fit + \
                novelty_alpha_2 * novelty_embedding_residual

            return novelty_embedding, treatmentEmbedding, torch.tensor(0., device=self.device)


class NRHUB_IVmediate_re_true(NRHUB_IVmediate):
    def __init__(self, config: NRHUB_Config):
        super().__init__(config)
        self.novelty_alpha_1 = MLP(config.alpha_1)
        self.novelty_alpha_2 = MLP(config.alpha_2)

        self.initParameter()

    def getItemEmbedding(self, treatment):
        # treatment: (batch_size, seq_len)
        ivEmbedding = self.itemIVembedding(treatment.long()).to(
            self.device)  # (batch_size, seq_len,ivEmbeddingSize)
        # ivEmbedding = self.IVMlp(ivEmbedding)
        treatmentEmbedding = self.itemEmbedding(treatment.long()).to(
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
            alpha_input = torch.cat(
                [novelty_embedding, novelty_IV], dim=-1)
            novelty_alpha_1 = self.novelty_alpha_1(alpha_input)
            novelty_alpha_2 = self.novelty_alpha_2(alpha_input)
            novelty_embedding = novelty_alpha_1 * novelty_embedding_fit + \
                novelty_alpha_2 * novelty_embedding_residual

            return novelty_embedding, treatmentEmbedding, torch.tensor(0., device=self.device)
