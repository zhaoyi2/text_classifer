import torch
from torch import nn
import torch.nn.functional as F
import os
import time

class PreActConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, pad_size):
        super(PreActConv, self).__init__()
        self.act = nn.ReLU()
        self.conv = nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=kernel_size, padding=pad_size)

    def forward(self, inputs):

        out = self.act(inputs)
        out = self.conv(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, pool_stride, pad_size):
        super(ConvBlock, self).__init__()
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=pool_stride, padding=pad_size)
        self.pre_act_conv = PreActConv(in_channel, out_channel, kernel_size, pad_size)

    def multi_conv(self, inputs, num_layers=2):
        for _ in range(num_layers):
            inputs = self.pre_act_conv(inputs)
        return inputs

    def forward(self, inputs, num_layers=2):
        out = self.max_pool(inputs)
        out_f = self.multi_conv(out, num_layers)
        out = out + out_f
        return out


class DPCNN(nn.Module):
    def __init__(self, config):
        super(DPCNN, self).__init__()

        vectors = config['vectors']
        embed_size = config['embed'] if vectors is None else vectors.size()[1]

        n_vocab = config['n_vocab']
        num_classes = config['num_classes']
        nums_filters = config['nums_filters']
        kernel_size = config['kernel_size']
        pool_stride = config['pool_stride']
        pad_size = config['pad_size']

        self.nums_block = config['nums_block']

        if vectors is not None:
            self.embedding = nn.Embedding.from_pretrained(vectors, freeze=False)
        else:
            self.embedding = nn.Embedding(n_vocab, embed_size)

        self.region_conv = nn.Conv2d(in_channels=1, out_channels=nums_filters,
                                     kernel_size=(kernel_size, embed_size), padding=(1, 0))

        self.conv = PreActConv(nums_filters, nums_filters, kernel_size, pad_size)
        self.conv_block = ConvBlock(nums_filters, nums_filters, kernel_size, pool_stride, pad_size)
        self.dense = nn.Linear(nums_filters, num_classes)

    def forward(self, inputs):
        out = self.embedding(inputs)
        out = out.unsqueeze(1)
        out = self.region_conv(out).squeeze(-1)
        for _ in range(2):
            out = self.conv(out)
        for _ in range(self.nums_block):
            out = self.conv_block(out)
        out = nn.MaxPool1d(kernel_size=out.size(2))(out)
        out = out.squeeze()
        out = self.dense(out)

        return out


def one_hot(input, num_classes):
    input = input.view(-1, 1)
    output = torch.zeros([input.size(0), num_classes], dtype=torch.long)
    return output.scatter_(1, input, 1)


if __name__ == '__main__':
    pass
