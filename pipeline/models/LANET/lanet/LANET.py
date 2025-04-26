import torch
from torch import nn
import math

from ..transformer import Constants


def get_non_pad_mask(seq, pad_idx):
    # assert seq.dim() == 2
    return seq.ne(pad_idx).type(torch.float).unsqueeze(-1)


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
        self.pad_idx = num_types
        self.do_finetune = do_finetune
        self.positional_encoding = PositionalEncoding(2*emb_dim)
        self.cat_embedding = nn.Embedding(num_embeddings=num_types + 1, embedding_dim=emb_dim)
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / self.emb_dim) for i in range(self.emb_dim)],
            device=torch.device(self.device))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * emb_dim,
            nhead=nhead,
            dim_feedforward=2 * emb_dim,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            layer_norm_eps=layer_norm_eps,
            bias=bias)

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=encoder_num_layers)

        self.dropout = nn.Dropout(0.1)
        self.encoder_history = nn.Linear(2 * emb_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def temporal_enc(self, time):
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def temporal_enc2(self, time):
        result = time.unsqueeze(-1) / self.position_vec
        result[0::2] = torch.sin(result[0::2])
        result[1::2] = torch.cos(result[1::2])
        return result


    def forward(self, event_type, event_time, events, max_len, pad_mask):
        # event_type # batch_size x seq_len x num_types
        # non_pad_mask # batch_size x seq_len x 1
        # event_time # batch_size x seq_len

        if self.do_finetune and self.finetune_type == "coles":
            batch = []
            for ind, user_events in enumerate(events):
                user = torch.zeros((max_len, 2*self.emb_dim))
                pad_vec = self.cat_embedding(torch.tensor(self.pad_idx))
                user[:, ] = torch.concat((pad_vec, pad_vec))
                for i, seq in enumerate(user_events):
                    time_embed = self.temporal_enc2(torch.tensor(event_time[ind][i]))
                    user[i] = torch.concat((torch.mean(self.cat_embedding(torch.tensor(seq)), dim=0), time_embed))
                batch.append(user)
            batch = torch.stack(batch).to(self.device)
            encoded_seq = self.positional_encoding(batch)  # batch_size x max_len x embed_dim
            x_encoder_output = self.transformer_encoder(encoded_seq, src_key_padding_mask=pad_mask)
            return torch.sum(x_encoder_output * (1 - pad_mask).unsqueeze(-1), dim=1) / (1 - pad_mask).sum(
                dim=1).unsqueeze(-1)

        non_pad_mask = get_non_pad_mask(event_time, self.pad_idx)
        temp_enc = self.temporal_enc(event_time) * non_pad_mask
        b = event_time.shape[0]
        s = event_type.shape[1]
        x_cat_emb = self.cat_embedding(torch.arange(self.num_types, device=torch.device(self.device))).unsqueeze(0).expand(b * s, -1, -1)
        # bs x num_types x emb_dim
        aux_mask = ((1 - torch.triu(torch.ones(s, s, device=event_type.device), diagonal=1).T)
                    .unsqueeze(2)
                    .expand(-1,-1,self.num_types)
                    .transpose(1, 0))
        # s x s x num_types
        x = ((event_type.unsqueeze(1).expand(-1, s, -1, -1) * aux_mask.unsqueeze(0).expand(b, -1, -1, -1))
             .reshape(b*s,s,self.num_types))
        # b x s x s x num_types
        x_t_emb = torch.sum(
            temp_enc.unsqueeze(1).expand(-1, s, -1, -1).reshape(b * s, s, self.emb_dim).unsqueeze(-1).expand(-1, -1, -1,
                                                                                                             self.num_types).transpose(
                3, 2) * \
            x.unsqueeze(-1), dim=1)  # bs x num_types x emb_dim

        x_encoder_input = torch.cat([x_cat_emb, x_t_emb], dim=2)
        x_encoder_output = self.transformer_encoder(x_encoder_input)
        x_hist = self.dropout(x_encoder_output)
        x_hist = self.encoder_history(x_hist).squeeze(2).reshape(b, s, -1)
        x_hist = self.sigmoid(x_hist)
        return x_hist, non_pad_mask
