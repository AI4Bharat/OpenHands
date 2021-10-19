import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from .utils import AttentionBlock


class PositionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=float(config.layer_norm_eps)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        position_ids = self.position_ids[:, :seq_length]

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = x + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERT(nn.Module):
    """
    BERT decoder module. 

    Args:
        n_features (int): Number of features in the input.
        num_class (int): Number of class for classification.
        config (dict): Configuration set for BERT layer.
    
    """
    def __init__(self, n_features, num_class, config):
        """
        pooling_type -> ["max","avg","att","cls"]
        """
        super().__init__()
        self.cls_token = config["cls_token"]

        if self.cls_token:
            self.pooling_type = "cls"
            self.cls_param = nn.Parameter(torch.randn(config.hidden_size))
        else:
            self.pooling_type = config["pooling_type"]

        self.l1 = nn.Linear(in_features=n_features, out_features=config.hidden_size)

        self.embedding = PositionEmbedding(config)
        model_config = transformers.BertConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
        )
        self.layers = nn.ModuleList(
            [
                transformers.BertLayer(model_config)
                for _ in range(config.num_hidden_layers)
            ]
        )

        if self.pooling_type == "att":
            self.attn_block = AttentionBlock(config.hidden_size)

        # self.bert = transformers.BertModel(model_config)
        self.l2 = nn.Linear(in_features=config.hidden_size, out_features=num_class)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape: (batch_size, T, n_features)
        
        returns:
            torch.Tensor: logits for classification.
        """
        x = self.l1(x)
        if self.cls_token:
            cls_embed = self.cls_param.unsqueeze(0).repeat(x.shape[0], 1, 1)
            x = torch.cat((cls_embed, x), dim=1)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)[0]

        if self.pooling_type == "cls":
            x = x[:, 0]
        elif self.pooling_type == "max":
            x = torch.max(x, dim=1).values
        elif self.pooling_type == "avg":
            x = torch.mean(x, dim=1)
        elif self.pooling_type == "att":
            x = self.attn_block(x)

        x = F.dropout(x, p=0.2)
        x = self.l2(x)
        return x
