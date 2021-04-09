import torch
import torch.nn as nn
import torch.nn.functional as F
from CE_loss_am_softmax import CE_loss
import numpy as np
import logging

logger = logging.getLogger(__name__)

class HANConfig(object):
    embedding_dim = 200
    seq_length = 100
    num_classes = 98
    vocab_size = 1200

    num_layers = 4
    hidden_dim = 256
    rnn = 'lstm'
    attention_dim = 256
    learning_rate = 1e-3
    dropout = 0.2
    batch_size = 128
    num_epochs = 10

    print_per_batch = 10
    save_per_batch = 10


class Attention(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super(Attention, self).__init__()
        self.attention_dim = attention_dim
        self.dense = nn.Linear(in_features=hidden_dim, out_features=attention_dim)
        self.u = nn.Linear(in_features=attention_dim, out_features=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        seq_len = x.size(1)
        hidden = x.size(-1)
        x_ = x.view(-1, hidden)
        v = self.tanh(self.dense(x_))
        alpha = F.softmax(self.u(v), 1)
        output = alpha.view(-1, seq_len, 1).mul(x).sum(1)
        return output


class HAN(nn.Module):

    def __init__(self, config):
        super(HAN, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)

        self.ln1 = nn.LayerNorm(self.config.embedding_dim)
        self.ln2 = nn.LayerNorm(self.config.hidden_dim*2)

        dropout = 0
        if self.config.num_layers > 1:
            dropout = self.config.dropout

        #kernel LSTM/GRU
        self.rnn = nn.LSTM(input_size=self.config.embedding_dim,hidden_size=self.config.hidden_dim, num_layers=self.config.num_layers,
                           batch_first=True, bidirectional=True, dropout=dropout)
        if self.config.rnn == 'gru':
            self.rnn = nn.GRU(input_size=self.config.embedding_dim, hidden_size=self.config.hidden_dim, num_layers=self.config.num_layers,
                              batch_first=True, bidirectional=True, dropout=dropout)

        self.att = Attention(self.config.hidden_dim*2, self.config.attention_dim)

        #self.dense = nn.Linear(in_features=self.config.hidden_dim*2,out_features=self.config.num_classes)
        self.W = nn.Parameter(torch.empty((self.config.hidden_dim*2, self.config.num_classes), dtype=torch.float), requires_grad=True)
        nn.init.kaiming_uniform_(self.W)

    def forward(self, x, lens=None):

        x = self.embedding(x)
        x = self.ln1(x)

        if lens is not None:
            lens_s, idxs = torch.sort(lens, descending=True)
            x = x[idxs]
            x = nn.utils.rnn.pack_padded_sequence(x, lens_s, batch_first=True)
        x, _ = self.rnn(x)

        if lens is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            _, idxs_ori = torch.sort(idxs, descending=False)
            x = x[idxs_ori]

        x = self.att(x)
        x = self.ln2(x)

        w_norm = torch.norm(self.W, p=2, dim=1, keepdim=True)
        w_norm = self.W/w_norm

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = x/x_norm

        return torch.mm(x_norm, w_norm)

if __name__ == "__main__":

    #config
    hc = HANConfig()
    hc.rnn = 'lstm'
    hc.num_classes = 5

    #HAN model init
    han = HAN(hc)

    #cuda check
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    han.to(device)
    # loss
    ce_loss = CE_loss()

    #optimizer
    optimizer = torch.optim.Adam(han.parameters(), lr=1e-3)

    inputs = torch.randint(0, 100, [128, 100])
    labels = torch.empty(128, dtype=torch.long).random_(5)

    lens = torch.empty(128, dtype=torch.long).random_(1, 100)

    #train
    han.train()
    out = han(inputs, lens)

    optimizer.zero_grad()

    loss = ce_loss(out, labels)

    loss.backward()

    optimizer.step()

    print(loss.detach().numpy())






