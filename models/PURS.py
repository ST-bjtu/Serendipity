import torch
import torch.nn as nn
from models.basic_model import BasicModel
from utils.layers import MLP, FullyConnectedLayer, Attention, AttentionSequencePoolingLayer
from config import PURS_Config


def mean_shift(input_X, window_radius=0.2):
    X1 = torch.transpose(input_X, 1, 2).unsqueeze(1)
    X2 = input_X.unsqueeze(1)
    C = input_X

    def _mean_shift_step(C):
        C = C.unsqueeze(3)
        Y = torch.sum(torch.pow(((C-X1)/window_radius), 2), dim=2)
        gY = torch.exp(-Y)
        num = torch.sum(gY.unsqueeze(3)*X2, dim=2)
        denom = torch.sum(gY, dim=2, keepdim=True)
        C = num / denom
        return C

    def _mean_shift(i, C, max_diff):
        new_C = _mean_shift_step(C)
        max_diff = torch.sum(torch.pow(new_C-C, 2), dim=1).sqrt().max().item()
        return i+1, new_C, max_diff

    def _cond(max_diff):
        return max_diff > 1e-5
    n_updates, C, max_diff = _mean_shift(0, C, 1e10)
    while _cond(max_diff):
        n_updates, C, max_diff = _mean_shift(0, C, max_diff)
    return C


class PURS(BasicModel):
    def __init__(self, config: PURS_Config):
        super().__init__(config)
        if config.preTrainEmbed:
            self.itemEmbedding = nn.Embedding.from_pretrained(
                torch.load(config.itemEmbeddingPath), freeze=True)
            self.item_mlp = MLP(config.item_mlp)
        else:
            self.itemEmbedding = nn.Embedding(
                config.item_num, config.itemEmbeddingSize)

        self.emb_size = config.itemEmbeddingSize
        self.hidden_size = config.hidden_size
        self.long_memory_window = config.long_memory_window  # 50
        self.short_memory_window = config.short_memory_window  # 15

        self.rnn = nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size,
                          num_layers=config.num_layers, batch_first=True)

        self.long_seq_attention = Attention(
            signal_length=self.long_memory_window, hidden_dim=self.hidden_size, Dense_dim=self.long_memory_window)
        # self.long_seq_attention = seq_attention(
        #     self.hidden_size, self.long_memory_window)
        self.drop_out = nn.Dropout(0.1)
        self.relevance_mlp = FullyConnectedLayer(
            input_size=config.itemEmbeddingSize + config.hidden_size,
            hidden_unit=[80, 40, 1],
            bias=[True, True, True],
            batch_norm=True,
            finalActivation=None,
            activation="sigmoid",
            dice_dim=2,
        )

        self.unexp_mlp = FullyConnectedLayer(
            input_size=2*config.itemEmbeddingSize,
            hidden_unit=[self.hidden_size, 1],
            bias=[True, True],
            batch_norm=True,
            finalActivation=None,
            activation="sigmoid",
            dice_dim=2,
        )
        self.unexp_attention = AttentionSequencePoolingLayer(
            embedding_dim=config.itemEmbeddingSize)
        self.sigmoid = nn.Sigmoid()

        self.initParameter()

    def initParameter(self):
        # if self.config.dataset == "ml-1m":
        #     self.apply(self.init_weights)
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
        treatmentEmbedding = self.itemEmbedding(treatment.long()).to(
            self.device)  # (batch_size, seq_len,itemEmbeddingSize)
        if self.config.preTrainEmbed:
            treatmentEmbedding = self.item_mlp(treatmentEmbedding)
        return treatmentEmbedding

    def unexpectedness(self, user_history_embedding, target_item_embedding):
        center = mean_shift(user_history_embedding)
        unexp = torch.mean(center, axis=1)
        unexp = torch.linalg.norm(
            unexp - target_item_embedding.squeeze(), ord=2, dim=1)
        return torch.exp(-1.0 * unexp)*unexp

    def forward(self, user_history, target_item, novelty_value):
        user_history_mask = torch.where(
            user_history == 0, 1, 0).bool().to(self.device)
        batch_size, seq_len = user_history.size()
        lengths = (user_history != 0).sum(
            dim=1).to(self.device)  # [batch_size]
        user_history_embedding = self.getItemEmbedding(user_history)
        target_item_embedding = self.getItemEmbedding(target_item.unsqueeze(1))

        history_packed = torch.nn.utils.rnn.pack_padded_sequence(
            user_history_embedding, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # RNN
        long_output_packed, hidden = self.rnn(history_packed, None)
        long_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            long_output_packed, batch_first=True, total_length=self.long_memory_window)
        long_preference = self.long_seq_attention(
            long_output, mask=user_history_mask)
        long_preference = self.drop_out(long_preference)

        relevance = self.relevance_mlp(
            torch.cat([long_preference, target_item_embedding.squeeze()], dim=-1))

        begin_index = lengths - self.short_memory_window
        begin_index[begin_index < 0] = 0
        positions = torch.arange(0, self.long_memory_window).to(self.device)
        unexp_mask = (positions.unsqueeze(0).expand(batch_size, self.long_memory_window) <
                      begin_index.unsqueeze(1).expand(batch_size, self.long_memory_window))
        unexp_history_mask = user_history_mask.clone()
        unexp_history_mask[unexp_mask] = True

        unexp_factor = self.unexp_attention(
            target_item_embedding, user_history_embedding, unexp_history_mask).squeeze()
        unexp_factor_out = self.unexp_mlp(
            torch.cat([unexp_factor, target_item_embedding.squeeze()], dim=-1))

        unexp = self.unexpectedness(
            user_history_embedding, target_item_embedding).unsqueeze(-1)

        total_out = self.sigmoid(relevance + unexp_factor_out * unexp)

        return [total_out], torch.tensor(0., device=self.device)
