import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import AttentionBlock


class RNNClassifier(nn.Module):
    def __init__(
        self,
        n_features,
        num_class,
        rnn_type="GRU",
        hidden_size=512,
        num_layers=1,
        bidirectional=True,
        use_attention=False,
    ):
        super().__init__()
        self.use_attention = use_attention

        self.rnn = getattr(nn, rnn_type)(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )

        rnn_out_size = hidden_size * 2 if bidirectional else hidden_size
        if self.use_attention:
            self.attn_block = AttentionBlock(hidden_size=rnn_out_size)
        self.fc = nn.Linear(rnn_out_size, num_class)

    def forward(self, x):
        """
        x.shape: (T, batch_size, n_features)
        """
        self.rnn.flatten_parameters()

        # Batch first
        cnn_embeds = x.transpose(0, 1)
        out, _ = self.rnn(cnn_embeds)

        if self.use_attention:
            out = self.fc(self.attn_block(out))
        else:
            # out = torch.max(out, dim=1).values
            out = self.fc(out[:, -1, :])

        return out
