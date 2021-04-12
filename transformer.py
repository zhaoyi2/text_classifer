import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Feedforward(nn.Module):
    def __init__(self, in_feats, out_feats, dropout=0.2):

        super(Feedforward, self).__init__()
        self.dense_1 = nn.Linear(in_feats, out_feats)
        self.dense_2 = nn.Linear(out_feats, in_feats)
        self.ln = nn.LayerNorm(in_feats)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):

        out = self.ln(input)

        out = self.relu(self.dense_1(out))
        out = self.dense_2(out)

        out = self.dropout(out)

        out = input + out

        return out


class DotProductAttention(nn.Module):

    def __init__(self, in_dims, out_dims, device, dropout=0.2):
        super(DotProductAttention, self).__init__()
        self.W_Q = nn.Linear(in_dims, out_dims)
        self.W_K = nn.Linear(in_dims, out_dims)
        self.W_V = nn.Linear(in_dims, out_dims)
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([out_dims])).to(self.device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):

        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        QK = torch.matmul(Q, K.permute(0, 2, 1))/self.scale
        QK = self.dropout(QK)

        att = torch.softmax(QK, dim=-1)

        if mask is not None:
            att = att.masked_fill(mask == 0, 1e-9)

        out = torch.matmul(att, V)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, in_dims, n_heads, device, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        assert in_dims % n_heads == 0
        self.head_len = int(in_dims/n_heads)
        self.ln = nn.LayerNorm(in_dims)
        self.dense = nn.Linear(in_dims, in_dims)
        self.multiheadattlayers = nn.ModuleList([DotProductAttention(in_dims, self.head_len, device, dropout) for _ in range(n_heads)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, out_enc=None, mask=None):
        out = self.ln(input)

        query, key, val = out, out, out
        if out_enc is not None:
            key, val = out_enc, out_enc
        out = torch.cat([layer(query, key, val, mask) for layer in self.multiheadattlayers], dim=-1)

        out = self.dense(out)

        out = self.dropout(out)

        out = input + out

        return out


class PosEmbedding(nn.Module):
    """绝对和相对位置编码"""
    def __init__(self, max_len, embedd_size, device):
        super().__init__()
        self.max_len = max_len
        self.embedd_size = embedd_size
        self.embedding = nn.Embedding(self.max_len, self.embedd_size)
        self.deveice = device

    def relative(self, pos):
        assert self.embedd_size % 2 == 0

        batch_size = pos.size(0)

        div = torch.pow(10000.0, torch.arange(0, 2 * self.embedd_size, 2, dtype=torch.float) / self.embedd_size).to(
            self.deveice)

        pos_emb = torch.zeros([batch_size, self.max_len, self.embedd_size]).to(self.deveice)
        pos_emb[:, :, 0: int(self.embedd_size / 2)] = torch.sin(pos.float().unsqueeze(-1) / div[0: int(self.embedd_size / 2)])
        pos_emb[:, :, int(self.embedd_size / 2):] = torch.cos(pos.float().unsqueeze(-1) / div[int(self.embedd_size / 2):])

        return pos_emb

    def absolute(self, pos):
        pos_emb = self.embedding(pos)

        return pos_emb


class EncodeLayer(nn.Module):
    def __init__(self, in_dims, n_heads, device, dropout):
        super(EncodeLayer, self).__init__()
        self.ln = nn.LayerNorm(in_dims)
        self.multiheadatt = MultiHeadAttention(in_dims, n_heads, device, dropout)
        self.feedfordlayer = Feedforward(in_dims, in_dims, dropout)

    def forward(self, input, src_mask):

        out = self.multiheadatt(input, mask=src_mask)

        out = self.feedfordlayer(out)

        return out


class DecodeLayer(nn.Module):
    def __init__(self, in_dims, n_heads, device, dropout):
        super(DecodeLayer, self).__init__()
        self.ln = nn.LayerNorm(in_dims)
        self.multiheadatt_en = MultiHeadAttention(in_dims, n_heads, device, dropout)
        self.multiheadatt_de = MultiHeadAttention(in_dims, n_heads, device, dropout)
        self.feedfordlayer = Feedforward(in_dims, in_dims, dropout)

    def forward(self, in_dec, out_enc, trg_mask):
        
        out = self.multiheadatt_en(in_dec, mask=trg_mask)
        out = self.multiheadatt_de(out, out_enc=out_enc, mask=trg_mask)
        out = self.feedfordlayer(out)
        return out


class Encoder(nn.Module):
    def __init__(self, encoder_config, device):
        super(Encoder, self).__init__()
        self.config = encoder_config
        self.embedding = nn.Embedding(self.config['vocab_size'], self.config['embedd_size'])
        if self.config['vector'] is not None:
            self.embedding = nn.Embedding.from_pretrained(self.config['vector'], freeze=self.config['freeze'])
        self.pos_embedding = PosEmbedding(self.config['max_len'], self.config['embedd_size'], device)
        self.ln = nn.LayerNorm(self.config['d_model'])
        self.encode_layers = nn.ModuleList([EncodeLayer(self.config['d_model'], self.config['n_heads'], device, self.config['dropout']) for _ in range(self.config['encode_layers'])])
        self.scale = torch.sqrt(torch.FloatTensor([self.config['d_model']])).to(device)

    def forward(self, input, mask=None):
        out = self.embedding(input)*self.scale + self.pos_embedding.absolute(input)

        for layer in self.encode_layers:
            out = layer(out, mask)

        out = self.ln(out)

        return out


class Decoder(nn.Module):
    def __init__(self, decode_config, device):
        super(Decoder, self).__init__()
        self.config = decode_config
        if self.config['vector'] is not None:
            self.embedding = nn.Embedding.from_pretrained(self.config['vector'], freeze=self.config['freeze'])
        self.embedding = nn.Embedding(self.config['vocab_size'], self.config['embedd_size'])
        self.pos_embedding = PosEmbedding(self.config['max_len'], self.config['embedd_size'], device)
        self.ln = nn.LayerNorm(self.config['d_model'])
        self.deocde_layers = nn.ModuleList([DecodeLayer(self.config['d_model'], self.config['n_heads'], device, self.config['dropout']) for _ in range(self.config['decode_layers'])])
        self.dense = nn.Linear(self.config['d_model'], self.config['num_classes'])
        self.scale = torch.sqrt(torch.FloatTensor([self.config['d_model']])).to(device)

    def forward(self, in_dec, out_enc, mask=None):
        out = self.embedding(in_dec)*self.scale + self.pos_embedding.absolute(in_dec)

        for layer in self.deocde_layers:
            out = layer(out, out_enc, mask)

        out = self.ln(out)
        out = self.dense(out)

        return out


class Transformer(nn.Module):
    def __init__(self, encode_config, decode_config, device):
        super(Transformer, self).__init__()
        self.encode_config = encode_config
        self.decode_config = decode_config

        self.encoder = Encoder(self.encode_config, device)
        self.decoder = Decoder(self.decode_config, device)

        self.device = device

    def pad_mask(self, input, pad_idx):
        #input shape: batch_size*seq_len
        mask = (input != pad_idx)
        mask = mask.unsqueeze(-1)
        return mask

    def seq_mask(self, input, pad_idx):
        #input shape: batch_size*seq_len
        batch_size = input.size(0)
        seq_len = input.size(1)

        pad_mask = self.pad_mask(input, pad_idx)

        seq_mask = torch.ones([seq_len, seq_len])
        seq_mask = torch.triu(seq_mask, 1)
        seq_mask = (seq_mask == 0).unsqueeze(0).repeat(batch_size, 1, 1)

        return pad_mask & seq_mask

    def forward(self, in_enc, pad_idx_enc, in_dec, pad_idx_dec):
        mask_enc = self.pad_mask(in_enc, pad_idx_enc)
        out_enc = self.encoder(in_enc, mask_enc)

        mask_dec = self.seq_mask(in_dec, pad_idx_dec)
        out_dec = self.decoder(in_dec, out_enc, mask_dec)

        out_flat = out_dec.contiguous().view(-1, out_dec.size(-1))
        return out_flat

    def recognize(self, sentense, src_vocab, trg_vocab, max_len):

        if sentense is None or sentense.strip()=='':
            raise IOError
        self.eval()
        tokens = ['sos']+[char.lower() for char in sentense]+['eos']
        in_enc = [src_vocab.index(toke) if toke in src_vocab else src_vocab.index('<unk>') for toke in tokens]
        pad_idx_enc = src_vocab.index('<pad>')
        pad_idx_dec = trg_vocab.index('<pad>')

        mask_enc = self.pad_mask(in_enc, pad_idx_enc)

        with torch.no_grad():
            out_enc = self.encoder(in_enc, mask_enc)

        out_idxs = ['<sos>']

        for t in max_len:
            in_dec = torch.LongTensor(out_idxs).unsqueeze(0).to(self.device)
            mask_dec = self.seq_mask(in_dec, pad_idx_dec)
            with torch.no_grad():
                out_dec = self.decoder(in_dec, out_enc, mask_dec)

            out_idx = torch.argmax(out_dec, dim=-1)[:, -1].item()
            if out_idx == trg_vocab['<eos>']:
                break
            out_idxs.append(out_idx)

        out_tokens = [trg_vocab[idx] for idx in out_idxs[1:]]
        return out_tokens






if __name__ == '__main__':

    encode_config = {
        'vector': None,
        'vocab_size': 8,
        'embedd_size': 200,
        'd_model': 200,
        'n_heads': 8,
        'freeze': True,
        'max_len': 100,
        'encode_layers': 2,
        'dropout': 0.2
    }

    decode_config = {
        'vector': None,
        'vocab_size': 8,
        'embedd_size': 200,
        'num_classes': 8,
        'd_model': 200,
        'n_heads': 8,
        'freeze': True,
        'max_len': 100,
        'decode_layers': 2,
        'dropout': 0.2
    }

    encode_vocab = ['<unk>', '<pad>', '<sos>', '<eos>', '我', '爱', '编', '程']
    decode_vocab = ['<unk>', '<pad>', '<sos>', '<eos>', '我', '爱', '编', '程']

    pad_idx_enc = encode_vocab.index('<pad>')
    pad_idx_dec = decode_vocab.index('<pad>')

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    transformer = Transformer(encode_config, decode_config, device)

    in_enc = ['<sos>', '我', '爱', '编', '程', '<eos>', '<pad>', '<pad>', '<pad>', '<pad>']
    in_dec = ['<sos>', '我', '爱', '编', '程', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']
    trg_label = ['我', '爱', '编', '程', '<eos>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']

    in_enc_idx = torch.LongTensor([encode_vocab.index(i) for i in in_enc]).unsqueeze(0)
    in_dec_idx = torch.LongTensor([decode_vocab.index(i) for i in in_dec]).unsqueeze(0)
    label = torch.tensor([decode_vocab.index(i) for i in trg_label])

    ce = nn.CrossEntropyLoss(ignore_index=pad_idx_dec)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)
    optimizer.zero_grad()

    out = transformer(in_enc_idx, pad_idx_enc, in_dec_idx, pad_idx_dec)
    print(out.shape)
    loss = ce(out, label)
    loss.backward()
    optimizer.step()
    print(loss.item())


