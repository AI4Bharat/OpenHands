import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    # Ref: https://github.com/0aqz0/SLR/blob/a1fc68b0ab4f3198efe767bc99745b9d31a13b0c/models/Attention.py

    def __init__(self, hidden_size):
        super(AttentionBlock, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        score_first_part = self.fc1(hidden_states)  # (B, T, H)
        h_t = hidden_states[:, -1, :]  # (B,H)
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)  # (B,T)
        attention_weights = F.softmax(score, dim=1)
        context_vector = torch.bmm(
            hidden_states.permute(0, 2, 1), attention_weights.unsqueeze(2)
        ).squeeze(
            2
        )  # (B,H)
        pre_activation = torch.cat((context_vector, h_t), dim=1)  # (B, H*2)
        attention_vector = self.fc2(pre_activation)  # (B, H)
        attention_vector = torch.tanh(attention_vector)

        return attention_vector
