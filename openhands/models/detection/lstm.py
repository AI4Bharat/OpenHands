import torch
import torch.nn as nn

# Paper: https://arxiv.org/abs/2008.04637
# Models ported from: https://github.com/google-research/google-research/tree/master/sign_language_detection

class SignDetectionRNN(nn.Module):
    def __init__(self, rnn_type="LSTM", input_dim=25, input_dropout = 0.5, num_layers = 1, hidden_size = 2**6, bidirectional = False):
        super().__init__()
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.rnn = getattr(nn, rnn_type)(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.hidden = None
        self.linear = nn.Linear(hidden_size, 2)
    
    def forward(self, x, hidden=None):
        # input.shape = [batch_size, seq_length, input_dim]
        x = self.input_dropout(x)
        rnn_out, self.hidden = self.rnn(x, hidden)
        # shape = [batch_size, out_dim]
        y_pred = self.linear(rnn_out)
        return y_pred#, self.hidden

