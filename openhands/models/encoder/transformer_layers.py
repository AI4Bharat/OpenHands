import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class SwiGLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear_g = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return F.silu(self.linear(x)) * self.linear_g(x)


class GEGLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear_g = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return F.gelu(self.linear(x)) * self.linear_g(x)


def get_activation(activation, in_dim=None, out_dim=None):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "swiglu":
        return SwiGLU(in_dim, out_dim)
    elif activation == "geglu":
        return GEGLU(in_dim, out_dim)
    else:
        raise NotImplementedError()


class LearnedPositionalEmbeddings(nn.Module):
    def __init__(self, dim, max_seq=512):
        super().__init__()
        self.d_model = dim
        self.max_seq = max_seq
        self.embed = nn.Embedding(max_seq, dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x, seq_dim=1):
        ids = torch.arange(x.shape[seq_dim], device=x.device)
        return self.embed(ids)[None, ...]


class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_dim=1):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, ...]


def apply_rotary_pos_emb(layer, sinusoidal_pos):
    def rotate_half(x):
        x = x.reshape(*x.shape[:-1], 2, -1)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    seq_len = layer.shape[-2]
    sinusoidal_pos = sinusoidal_pos[:, :, -seq_len:]
    return (layer * sinusoidal_pos.cos()) + (rotate_half(layer) * sinusoidal_pos.sin())


class RelativePositionEncodingBias(nn.Module):
    """
    Adopted from https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py
    """

    def __init__(
        self, num_heads=2, num_buckets=32, max_distance=128, bidirectional=True
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(
                torch.long
            ) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = (
            memory_position - context_position
        )  # (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # (query_length, key_length)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
        )
        relative_position_bucket = relative_position_bucket.to(
            self.relative_attention_bias.weight.device
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # (1, num_heads, query_length, key_length)
        return values

    def forward(self, query_length, key_length):
        return self.compute_bias(query_length, key_length)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8, bias=True):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d_model))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d_model))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        x_normed = x / (norm * self.d_model ** -0.5 + self.eps)
        x_scaled = x_normed * self.scale

        if self.bias:
            x_scaled += self.offset

        return x_scaled


############################################


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must divisible by num_heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, pos_bias=None, sinusoidal_pos=None, mask=None):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        if sinusoidal_pos is not None:
            q = apply_rotary_pos_emb(q, sinusoidal_pos)
            k = apply_rotary_pos_emb(k, sinusoidal_pos)

        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))

        if pos_bias is not None:
            attn_logits += pos_bias

        attn_logits = attn_logits / math.sqrt(d_k)

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)

        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        out = self.o_proj(values)

        return out


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads,
        feedforward_dim,
        dropout=0.0,
        activation="gelu",
        norm_type="layernorm",
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(input_dim, input_dim, num_heads)

        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim, feedforward_dim),
            nn.Dropout(dropout),
            get_activation(activation, feedforward_dim, feedforward_dim),
            nn.Linear(feedforward_dim, input_dim),
        )

        if norm_type == "layernorm":
            self.norm1 = nn.LayerNorm(input_dim)
            self.norm2 = nn.LayerNorm(input_dim)
        elif norm_type == "rmsnorm":
            self.norm1 = RMSNorm(input_dim)
            self.norm2 = RMSNorm(input_dim)
        else:
            raise NotImplementedError()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, pos_bias=None, sinusoidal_pos=None):

        # TODO: make norm position as param???
        # pre-norm
        x = self.norm1(x)
        attn_out = self.self_attn(
            x, pos_bias=pos_bias, sinusoidal_pos=sinusoidal_pos, mask=mask
        )
        x = x + self.dropout(attn_out)

        x = self.norm2(x)
        linear_out = self.linear_layers(x)
        x = x + self.dropout(linear_out)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads,
        num_layers,
        feedforward_dim,
        dropout=0,
        activation="gelu",
        pos_embed_type="sinusoidal",
        norm_type="rmsnorm",
        max_pos_embeddings=512,
    ):
        """
        Transformer layer with suggestions proposed in https://arxiv.org/pdf/2102.11972.pdf

        embed_type => {learned, sinusoidal, relative_bias, relative_bias_shared, rotary}
        norm_type => {layernorm, rmsnorm}
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout
        self.max_pos_embeddings = max_pos_embeddings

        self.pos_embed_type = pos_embed_type
        self.norm_type = norm_type

        if self.pos_embed_type == "learned":
            self.embed_layer = LearnedPositionalEmbeddings(
                self.input_dim, max_pos_embeddings
            )
        elif self.pos_embed_type == "sinusoidal":
            self.embed_layer = SinusoidalPositionalEmbeddings(self.input_dim)
        elif (
            self.pos_embed_type == "relative_bias"
            or self.pos_embed_type == "relative_bias_shared"
        ):
            self.rel_pos = RelativePositionEncodingBias(num_heads)

        elif self.pos_embed_type == "rotary":
            self.embed_layer = SinusoidalPositionalEmbeddings(
                self.input_dim // self.num_heads
            )
        else:
            raise NotImplementedError()

        self.blocks_list = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    self.input_dim,
                    self.num_heads,
                    self.feedforward_dim,
                    activation=activation,
                    norm_type=norm_type,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x, mask=None):
        seq_length = x.size(1)

        if self.pos_embed_type == "learned" or self.pos_embed_type == "sinusoidal":
            position_embed = self.embed_layer(x)
            x = x + position_embed

        sinusoidal_pos = None
        pos_bias = None

        if (
            self.pos_embed_type == "relative_bias_shared"
            or self.pos_embed_type == "relative_bias"
        ):
            # use the same bias for all layers in shared
            pos_bias = self.rel_pos(seq_length, seq_length)

        elif self.pos_embed_type == "rotary":
            sinusoidal_pos = self.embed_layer(x)[None, ...]

        for ind, block in enumerate(self.blocks_list):
            x = block(x, mask=mask, pos_bias=pos_bias, sinusoidal_pos=sinusoidal_pos)

            if self.pos_embed_type == "relative_bias":
                # change pos_bias for each block
                pos_bias = self.rel_pos(seq_length, seq_length)

        return x
