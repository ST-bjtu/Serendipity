import torch
import torch.nn as nn
import numpy as np
from config import SNPR_Config
from utils.layers import TransformerLayer, FullyConnectedLayer, MLP

from models.basic_model import BasicModel


class additiveAttention(nn.Module):
    def __init__(self, embedding_dim) -> None:
        super().__init__()
        self.linear1 = nn.Linear(
            in_features=embedding_dim*2, out_features=embedding_dim, bias=False)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(
            in_features=embedding_dim, out_features=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, user_behavior, mask=None):
        # query [batch_size, 1, embed_size]
        # user_behavior [batch_size, seq_len, embed_size]
        # mask [batch_size, seq_len]
        user_behavior_len = user_behavior.size(1)
        queries = query.expand(-1, user_behavior_len, -1)
        attention_input = torch.cat([queries, user_behavior], dim=-1)

        # [batch_size, seq_len, 1]
        attention_score = self.linear2(
            self.tanh(self.linear1(attention_input)))
        attention_score = torch.transpose(
            attention_score, 1, 2)  # [batch_size, 1, seq_len]

        if mask is not None:
            attention_score = attention_score.masked_fill(
                mask.unsqueeze(1), torch.tensor(0))

        attention_score = self.softmax(attention_score)

        output = torch.matmul(attention_score, user_behavior)

        return output.squeeze()


class SNPR(BasicModel):
    def __init__(self, config: SNPR_Config):
        super().__init__(config)
        self.item_num = config.item_num
        self.emb_size = config.itemEmbeddingSize
        # self.max_his = config.pad_len + 1
        self.max_his = config.pad_len
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.len_range = torch.from_numpy(
            np.arange(self.max_his)).to(self.device)

        if self.config.preTrainEmbed:
            self.i_embeddings = nn.Embedding.from_pretrained(
                torch.load(config.itemEmbeddingPath), freeze=True)
            self.item_mlp = MLP(self.config.item_mlp)
        else:
            self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)

        self.transformer_block = nn.ModuleList([
            TransformerLayer(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.num_heads,
                             dropout=self.dropout, kq_same=False, activation='gelu')
            for _ in range(self.num_layers)
        ])

        self.attention = additiveAttention(embedding_dim=self.emb_size)

        self.unexp_linear = nn.Linear(self.emb_size, self.emb_size)
        self.gelu = nn.GELU()

        self.unexp_fc_layer = FullyConnectedLayer(input_size=self.emb_size, hidden_unit=[
                                                  64, 32, 1], bias=[True, True, True], batch_norm=False)
        self.relevance_fc_layer = FullyConnectedLayer(
            input_size=self.emb_size*2, hidden_unit=[64, 32, 1], bias=[True, True, True], batch_norm=False)

        self.trade_off = config.trade_off

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

    def transform_encoder(self, history):
        batch_size, seq_len = history.shape
        lengths = (history != 0).sum(dim=1).to(self.device)  # [batch_size]

        valid_his = (history > 0).long().to(self.device)
        his_vectors = self.getItemEmbedding(history)

        # Position embedding
        # lengths:  [4, 2, 5]
        # position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
        position = (lengths[:, None] -
                    self.len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position.cpu()).to(self.device)
        his_vectors = his_vectors + pos_vectors
        # Self-attention
        causality_mask = np.tril(
            np.ones((1, 1, seq_len, seq_len), dtype=np.int))
        attn_mask = torch.from_numpy(causality_mask).to(self.device)
        for block in self.transformer_block:
            his_vectors = block(his_vectors, attn_mask)
        # [batch_size, seq_len, embed_size]
        his_vectors = his_vectors * valid_his[:, :, None].float()

        return his_vectors

    def getNewHistory(self, history, target_item):
        batch_size, seq_len = history.size()
        new_history = torch.cat(
            [history, torch.zeros(batch_size, 1)], dim=-1).long()
        new_lengths = (new_history != 0).sum(dim=1)
        new_len_range = torch.from_numpy(np.arange(seq_len + 1))
        insert_mask = (new_len_range[None, :] == new_lengths[:, None])
        new_history[insert_mask] = target_item.squeeze()

        return new_history

    def splitNewHistory(self, new_history, new_history_embed):
        batch_size, seq_len, embed_size = new_history_embed.size()
        new_lengths = (new_history != 0).sum(dim=1)
        init_len_range = torch.from_numpy(np.arange(seq_len - 1))

        history_gather_index = torch.cat(
            [init_len_range.clone().unsqueeze(0) for _ in range(batch_size)], dim=0)
        mask = (history_gather_index >= (new_lengths[:, None] - 1)).clone()
        new_value = (history_gather_index[mask].clone() + 1).clone()
        history_gather_index[mask] = new_value
        history_gather_index = history_gather_index.to(self.device)

        torch.use_deterministic_algorithms(False)
        target_gather_index = (new_lengths[:, None, None] - 1).to(self.device)
        target_item_embed = torch.gather(
            new_history_embed, 1, target_gather_index.expand(-1, -1, embed_size))

        init_history_embed = torch.gather(
            new_history_embed, 1, history_gather_index[:, :, None].expand(-1, -1, embed_size))

        torch.use_deterministic_algorithms(True)

        return target_item_embed, init_history_embed

    def forward(self, user_history, target_item, novelty_value):
        history = user_history.long()  # [batch_size, seq_len]
        history_embedding = self.getItemEmbedding(history)
        i_ids = target_item.long().unsqueeze(-1)  # [batch_size, -1]
        i_vectors = self.getItemEmbedding(i_ids)
        general_representations = torch.cat(
            [history_embedding, i_vectors], dim=1).mean(dim=1)  # [batch_size, embed_size]

        # new_history = self.getNewHistory(history, i_ids)
        new_history = history
        his_vectors = self.transform_encoder(new_history)

        history_mask = torch.where(
            history == 0, 1, 0).bool().to(self.device)

        # target_vector, history_vectors = self.splitNewHistory(
        #     new_history, his_vectors)
        target_vector = i_vectors
        history_vectors = his_vectors
        history_attention = self.attention(
            target_vector, history_vectors, mask=history_mask)  # [batch_size, embed_size]

        unexp_representation = self.gelu(
            self.unexp_linear(his_vectors.mean(dim=1)))  # [batch_size, embed_size]
        relevance_representation = torch.cat(
            [general_representations, history_attention], dim=-1)  # [batch_size, embed_size*2]

        unexp = self.unexp_fc_layer(unexp_representation)
        relevance = self.relevance_fc_layer(relevance_representation)

        click_score = self.trade_off * relevance + (1 - self.trade_off) * unexp

        return [click_score], torch.tensor(0., device=self.device)
