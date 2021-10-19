import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import AttentionBlock


class RNNClassifier(nn.Module):
    """
    RNN head for classification.
    
    Args:
        n_features (int): Number of features in the input.
        num_class (int): Number of class for classification.
        rnn_type (str): GRU or LSTM. Default: ``GRU``.
        hidden_size (str): Hidden dim to use for RNN. Default: 512.
        num_layers (int): Number of layers of RNN to use. Default: 1.
        bidirectional (bool): Whether to use bidirectional RNN or not. Default: ``True``.
        use_attention (bool): Whether to use attenion for pooling or not. Default: ``False``.

    """
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
        Args:
            x (torch.Tensor): Input tensor of shape: (batch_size, T, n_features)
        
        returns:
            torch.Tensor: logits for classification.
        """
        self.rnn.flatten_parameters()

        out, _ = self.rnn(x)

        if self.use_attention:
            out = self.fc(self.attn_block(out))
        else:
            # out = torch.max(out, dim=1).values
            out = self.fc(out[:, -1, :])

        return out
