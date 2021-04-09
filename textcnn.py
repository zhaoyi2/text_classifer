
import torch
from torch import nn
import torch.nn.functional as F
import os
import time


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        vectors = config['vectors']
        embed_size = config['embed'] if vectors is None else vectors.size()[1]
        n_vocab = config['n_vocab']
        dropout = config.get('dropout', 0)
        num_classes = config['num_classes']
        num_filters = config['num_filters']
        filter_sizes = config['filter_sizes']
        if vectors is not None:
            self.embedding = nn.Embedding.from_pretrained(vectors, freeze=False)
        else:
            self.embedding = nn.Embedding(n_vocab, embed_size)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed_size)) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        x = torch.relu(conv(x))
        x = x.squeeze(3)
        x = F.max_pool1d(x, int(x.size(2)))
        x = x.squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        print(out.shape)
        return out


if __name__ == '__main__':
    pass