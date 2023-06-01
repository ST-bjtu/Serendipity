import torch
import torch.nn as nn
import numpy as np


class Dice(nn.Module):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.
    Input shape:
        - 2 dims: [batch_size, embedding_size(features)]
        - 3 dims: [batch_size, num_features, embedding_size(features)]
    Output shape:
        - Same shape as input.
    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
        - https://github.com/zhougr1993/DeepInterestNetwork, https://github.com/fanoping/DIN-pytorch
    """

    def __init__(self, emb_size, dim=2, epsilon=1e-8):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3

        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        # wrap alpha in nn.Parameter to make it trainable
        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((emb_size,)))
        else:
            self.alpha = nn.Parameter(torch.zeros((emb_size, 1)))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        return out


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_unit, bias, batch_norm=False, activation='relu', finalActivation='sigmoid', dice_dim=2):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_unit) >= 1 and len(bias) >= 1
        assert len(bias) == len(hidden_unit)
        self.finalActivation = finalActivation

        layers = []
        layers.append(nn.Linear(input_size, hidden_unit[0], bias=bias[0]))

        for i, h in enumerate(hidden_unit[:-1]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_unit[i]))

            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'dice':
                assert dice_dim
                layers.append(Dice(hidden_unit[i], dim=dice_dim))
            elif activation.lower() == 'prelu':
                layers.append(nn.PReLU())
            elif activation.lower() == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise NotImplementedError

            layers.append(
                nn.Linear(hidden_unit[i], hidden_unit[i+1], bias=bias[i]))

        self.fc = nn.Sequential(*layers)
        if finalActivation == 'sigmoid':
            self.output_layer = nn.Sigmoid()
        elif finalActivation == 'leakyRelu':
            self.output_layer = nn.LeakyReLU()
        elif finalActivation == 'relu':
            self.output_layer = nn.ReLU()
        elif finalActivation == 'tanh':
            self.output_layer = nn.Tanh()
        elif finalActivation == None:
            self.output_layer = None
        elif finalActivation == 'none':
            self.output_layer = None
        else:
            raise ValueError("activate function error!")

        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.output_layer is not None else self.fc(x)


class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, embedding_dim=4):
        super(AttentionSequencePoolingLayer, self).__init__()

        # TODO: DICE acitivation function
        # TODO: attention weight normalization
        self.local_att = LocalActivationUnit(hidden_unit=[64, 16], bias=[
                                             True, True], embedding_dim=embedding_dim, batch_norm=False)

    def forward(self, query_ad, user_behavior, mask=None):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        # mask                : size -> batch_size * time_seq_len
        # output              : size -> batch_size * 1 * embedding_size

        attention_score = self.local_att(query_ad, user_behavior)
        attention_score = torch.transpose(attention_score, 1, 2)  # B * 1 * T

        if mask is not None:
            attention_score = attention_score.masked_fill(
                mask.unsqueeze(1), torch.tensor(0))

        # multiply weight
        output = torch.matmul(attention_score, user_behavior)

        return output


class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_unit=[80, 40], bias=[True, True], embedding_dim=4, batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4*embedding_dim,
                                       hidden_unit=hidden_unit,
                                       bias=bias,
                                       batch_norm=batch_norm,
                                       finalActivation=None,
                                       activation='dice',
                                       dice_dim=3)

        self.fc2 = nn.Linear(hidden_unit[-1], 1)

    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size

        user_behavior_len = user_behavior.size(1)

        queries = query.expand(-1, user_behavior_len, -1)

        attention_input = torch.cat([queries, user_behavior, queries-user_behavior, queries*user_behavior],
                                    dim=-1)  # as the source code, subtraction simulates verctors' difference

        attention_output = self.fc1(attention_input)
        attention_score = self.fc2(attention_output)  # [B, T, 1]

        return attention_score


class Attention(nn.Module):
    def __init__(self, signal_length, hidden_dim, Dense_dim):
        super().__init__()
        self.signal_length = signal_length
        self.hidden_dim = hidden_dim
        self.dense_dim = Dense_dim
        self.Uq = nn.Linear(self.hidden_dim, self.dense_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.vq = nn.Parameter(torch.randn(1, 1, self.dense_dim))

    def reset_parameters(self):
        nn.init.normal_(self.vq)
        nn.init.normal_(self.Uq)

    def forward(self, x, mask=None):
        """
        Args:
            x: [bs, sl, hd]
        Return
            res: [bs, hd]
        """
        # bs, sl ,dd
        key = self.tanh(self.Uq(x))

        # bs, 1, sl
        score = self.vq.matmul(key.transpose(-2, -1))
        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(1), torch.tensor(-1e6))

        weight = self.softmax(score)
        res = weight.matmul(x).sum(dim=1)
        # bs, hd
        return res


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "MLP"

        self.hidden_dims = config['hidden_dims']
        self.droput = config['dropout']
        self.is_dropout = config['is_dropout']

        self.activation = nn.Tanh()

        for i in range(1, len(self.hidden_dims)):
            setattr(self, "linear_"+str(i),
                    nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
            if config["is_dropout"]:
                setattr(self, 'dropout_'+str(i),
                        nn.Dropout(self.droput[i-1]))

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(1, len(self.hidden_dims)):
            nn.init.xavier_uniform_(getattr(self, "linear_"+str(i)).weight)

    def forward(self, x):
        deep_out = x
        for i in range(1, len(self.hidden_dims)):
            deep_out = getattr(self, 'linear_'+str(i))(deep_out)
            deep_out = self.activation(deep_out)
            if self.is_dropout:
                deep_out = getattr(self, "dropout_"+str(i))(deep_out)
        return deep_out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=False, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention.
        """
        self.d_model = d_model
        self.h = n_heads
        self.d_k = self.d_model // self.h
        self.kq_same = kq_same

        if not kq_same:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)

    def head_split(self, x):  # get dimensions bs * h * seq_len * d_k
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        return x.view(*new_x_shape).transpose(-2, -3)

    def forward(self, q, k, v, mask=None):
        origin_shape = q.size()

        # perform linear operation and split into h heads
        if not self.kq_same:
            q = self.head_split(self.q_linear(q))
        else:
            q = self.head_split(self.k_linear(q))
        k = self.head_split(self.k_linear(k))
        v = self.head_split(self.v_linear(v))

        # calculate attention using function we will define next
        output = self.scaled_dot_product_attention(q, k, v, self.d_k, mask)

        # concatenate heads and put through final linear layer
        output = output.transpose(-2, -3).reshape(origin_shape)
        return output

    @staticmethod
    def scaled_dot_product_attention(q, k, v, d_k, mask=None):
        """
        This is called by Multi-head attention object to find the values.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / \
            d_k ** 0.5  # bs, head, q_len, k_len
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)
        scores = (scores - scores.max()).softmax(dim=-1)
        scores = scores.masked_fill(torch.isnan(scores), 0)
        output = torch.matmul(scores, v)  # bs, head, q_len, d_k
        return output


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout=0, kq_same=False, activation='relu'):
        super().__init__()
        """
        This is a Basic Block of Transformer. It contains one Multi-head attention object. 
        Followed by layer norm and position wise feedforward net and dropout layer.
        """
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, n_heads, kq_same=kq_same)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        if activation == 'relu':
            self.activate = nn.ReLU()
        elif activation == 'gelu':
            self.activate = nn.GELU()
        else:
            raise ValueError("activate error!")

    def forward(self, seq, mask=None):
        context = self.masked_attn_head(seq, seq, seq, mask)
        context = self.layer_norm1(self.dropout1(context) + seq)
        output = self.activate(self.linear1(context))
        output = self.linear2(output)
        output = self.layer_norm2(self.dropout2(output) + context)
        return output
