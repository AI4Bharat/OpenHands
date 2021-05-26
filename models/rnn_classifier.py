import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

#Ref: https://github.com/0aqz0/SLR/blob/a1fc68b0ab4f3198efe767bc99745b9d31a13b0c/models/Attention.py

class AttentionBlock(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionBlock, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        
    def forward(self, hidden_states):
        # (batch_size, time_steps, hidden_size)
        score_first_part = self.fc1(hidden_states)
        # (batch_size, hidden_size)
        h_t = hidden_states[:,-1,:]
        # (batch_size, time_steps)
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(score, dim=1)
        # (batch_size, hidden_size)
        context_vector = torch.bmm(hidden_states.permute(0,2,1), attention_weights.unsqueeze(2)).squeeze(2)
        # (batch_size, hidden_size*2)
        pre_activation = torch.cat((context_vector, h_t), dim=1)
        # (batch_size, hidden_size)
        attention_vector = self.fc2(pre_activation)
        attention_vector = torch.tanh(attention_vector)

        return attention_vector

class ConvRNNClassifier(nn.Module):
    def __init__(self,num_class, backbone='resnet18', pretrained=True, rnn_type="GRU", rnn_hidden_size=512, rnn_num_layers=1, bidirectional=True, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        n_out_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.rnn = getattr(nn, rnn_type)(input_size=n_out_features, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, bidirectional=bidirectional)
        
        rnn_out_size = rnn_hidden_size*2 if bidirectional else rnn_hidden_size
        if self.use_attention:
            self.attn_block = AttentionBlock(hidden_size=rnn_out_size)
        self.fc = nn.Linear(rnn_out_size, num_class)
        
    def forward(self, x):
        b, c, t, h, w = x.shape
        cnn_embeds = []
        for i in range(t):
            out = self.backbone(x[:,:, i, :, :]) 
            out = out.view(out.shape[0], -1)
            cnn_embeds.append(out)
        
        cnn_embeds = torch.stack(cnn_embeds, dim=0)
        #Batch first
        self.rnn.flatten_parameters()
        cnn_embeds = cnn_embeds.transpose(0,1)
        out, _ = self.rnn(cnn_embeds)
        
        out = self.fc(out[:, -1, :])
        return out