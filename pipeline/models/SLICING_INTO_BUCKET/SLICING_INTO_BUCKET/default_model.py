import torch 
from torch import nn
import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size,
                 maxlen=5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        device = token_embedding.device
        return token_embedding + self.pos_embedding[:token_embedding.size(1), :].detach().to(device)


class TransformerEncoder(nn.Module):
    def __init__(self, 
                 num_types, 
                 emb_dim, 
                 nhead=4,
                 dropout=0.2,
                 activation='relu',
                 batch_first=True,
                 encoder_num_layers=2,
                 layer_norm_eps=1e-5,
                 bias=True,
                 device="cpu",
                 do_finetune=False,
                 finetune_type="no"
                 ):
        
        super(TransformerEncoder, self).__init__()

        self.num_types = num_types
        self.emb_dim = emb_dim
        self.finetune_type = finetune_type
        self.device = device
        self.do_finetune = do_finetune
        self.pad_idx = num_types
        self.positional_encoding = PositionalEncoding(emb_dim)
        self.cat_embedding = nn.Embedding(num_embeddings=num_types + 1, embedding_dim=emb_dim, padding_idx=self.pad_idx)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=2*emb_dim,
            dropout=dropout,
            activation=activation, 
            batch_first=batch_first, 
            layer_norm_eps=layer_norm_eps,
            bias=bias)
        
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=encoder_num_layers)
        self.linear = nn.Linear(emb_dim, self.num_types)

    def forward(self, events, max_len, future_mask, pad_mask):
        if self.do_finetune:
            encoded_seq = self.cat_embedding(events)
        else:
            batch = []
            for ind, user_events in enumerate(events):
                user = torch.zeros((max_len, self.emb_dim))
                pad_vec = self.cat_embedding(torch.tensor(self.pad_idx))
                user[:, ] = pad_vec
                for i, seq in enumerate(user_events):
                    user[i] = torch.mean(self.cat_embedding(torch.tensor(seq)), dim=0)
                batch.append(user)
            batch = torch.stack(batch).to(self.device)
            encoded_seq = self.positional_encoding(batch)  # batch_size x max_len x embed_dim
        if self.do_finetune and self.finetune_type == "coles":
            x_encoder_output = self.transformer_encoder(encoded_seq, src_key_padding_mask=pad_mask)
            return torch.sum(x_encoder_output * (1 - pad_mask).unsqueeze(-1), dim=1) / (1 - pad_mask).sum(dim=1).unsqueeze(-1)

        x_encoder_output = self.transformer_encoder(encoded_seq, future_mask, pad_mask)
        x_hist = self.linear(x_encoder_output)  # batch_size x max_len x num_types
        return x_hist
