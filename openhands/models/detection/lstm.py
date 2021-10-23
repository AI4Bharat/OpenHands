import torch
import torch.nn as nn

# Paper: https://arxiv.org/abs/2008.04637
# Models ported from: https://github.com/google-research/google-research/tree/master/sign_language_detection

class SignDetectionLSTM(nn.Module):
    def __init__(self, input_dim=25, input_dropout = 0.5, encoder_layers = 1, hidden_size = 2**6, encoder_bidirectional = False):
        super(SignDetectionLSTM, self).__init__()
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.lstm = nn.LSTM(input_dim, hidden_size, encoder_layers, bidirectional=encoder_bidirectional, batch_first=True)
        self.linear = nn.Linear(hidden_size, 2)
    
    def forward(self, x, hidden=None):
        # input.shape = [batch_size, seq_length, input_dim]
        x = self.input_dropout(x)
        lstm_out, self.hidden = self.lstm(input, hidden)
        # shape = [batch_size, out_dim]
        y_pred = self.linear(self.dropout(lstm_out[:,-1,:]))
        return y_pred

