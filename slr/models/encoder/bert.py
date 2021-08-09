import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class BertModel(nn.Module):
    def __init__(self, n_features: int, config, trainable_positional_embeddings=False):
        super(BertModel, self).__init__()

        self.embed_layer = nn.Linear(n_features, config.hidden_size)
        if trainable_positional_embeddings:
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
        else:
            self.position_embeddings = PositionalEncoding(
                config.hidden_size, max_len=config.max_position_embeddings
            )
        self.embedding_layer_norm = nn.LayerNorm(config.hidden_size)
        self.embedding_dropout = nn.Dropout(p=config.hidden_dropout_prob)

        self.encoders = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.attention_probs_dropout_prob,
                activation=config.hidden_act,
                batch_first=True,
            ),
            num_layers=config.num_hidden_layers,
        )
        self.n_out_features = config.hidden_size

    def forward(self, input_):
        if len(input_.shape) == 5:
            B, C, T, V, M = input_.shape
            # Convert to B,T,V*C
            input_ = input_.permute(0, 2, 1, 3, 4).reshape(B, T, C * V * M)
            # TODO: Generalize the format for all encoders

        coordinate_embeddings = self.embed_layer(input_)
        embeddings = self.position_embeddings(coordinate_embeddings)

        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        encoder_outputs = self.encoders(embeddings)
        return encoder_outputs
