#!/usr/bin/env python
# encoding: utf-8
import torch
from torch import nn
import torch.functional as F


class JointCut(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.disable_type_embed = cfg.disable_type_embed if 'disable_type_embed' in cfg else False
        if not self.disable_type_embed:
            self.transformer_d_model = cfg.char_embed_dim + cfg.type_embed_dim
            self.hidden_dim = cfg.n_gram * (self.transformer_d_model + cfg.type_embed_dim)
        else:
            self.transformer_d_model = cfg.char_embed_dim
            self.hidden_dim = cfg.n_gram * self.transformer_d_model

        self.char_embed = nn.Embedding(cfg.char_vocab_size, cfg.char_embed_dim)
        self.char_embed_drop = nn.Dropout(cfg.char_embed_dropout)

        self.type_embed = nn.Embedding(cfg.type_vocab_size, cfg.type_embed_dim)

        # self.pos_encoder = PositionalEncoding(cfg.char_embed_dim + cfg.type_embed_dim, cfg.n_gram, cfg.device)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.transformer_d_model,
                                       nhead=cfg.transformer_n_head,
                                       dim_feedforward=cfg.transformer_dim_feedforward,
                                       activation=cfg.transformer_activation),
            num_layers=cfg.transformer_num_layers)

        self.hidden_drop = nn.Dropout(cfg.hidden_dropout)

        self.hidden_activation = cfg.hidden_activation

        self.syllable_linear = nn.Linear(self.hidden_dim, cfg.syllable_dense_dim)
        self.syllable_cls = nn.Linear(cfg.syllable_dense_dim, 1)

        self.word_linear = nn.Linear(self.hidden_dim, cfg.word_dense_dim)
        self.word_cls = nn.Linear(cfg.word_dense_dim + cfg.syllable_dense_dim, 1)

        self.syllable_loss_lambda = cfg.syllable_loss_lambda
        self.align_loss_lambda = cfg.align_loss_lambda

        self.syllable_loss = nn.BCELoss()
        self.word_loss = nn.BCELoss()
        self.align_loss = nn.MSELoss(reduction="none")

        self.swish = torch.nn.SiLU()

    def _get_activation(self, act):
        activations = {'gelu': torch.nn.GELU(), 'relu': torch.relu, 'tanh': torch.tanh}
        return activations[act]

    def forward(self, in_char, in_type):
        char_embed = self.char_embed(in_char.t())
        char_embed = self.char_embed_drop(char_embed)

        type_embed = self.type_embed(in_type.t())

        if self.disable_type_embed:
            embed = char_embed
        else:
            embed = torch.cat([char_embed, type_embed], dim=-1)

        # embed = self.pos_encoder(embed)

        hidden = self.transformer_encoder(embed)

        if not self.disable_type_embed:
            hidden = torch.cat((hidden, type_embed), dim=-1)

        hidden = self.hidden_drop(hidden)

        hidden_act = self._get_activation(self.hidden_activation)

        hidden = hidden.permute(1, 0, 2).contiguous().view(in_char.size(0), -1)
        syllable_hidden = hidden_act(self.syllable_linear(hidden))

        word_hidden = hidden_act(self.word_linear(hidden))
        word_hidden = torch.cat([word_hidden, syllable_hidden], dim=-1)

        word_logits = torch.sigmoid(self.word_cls(word_hidden).view(-1))
        syllable_logits = torch.sigmoid(self.syllable_cls(syllable_hidden).view(-1))

        return word_logits, syllable_logits

    def joint_loss(self, word_logits, word_labels, syllable_logits, syllable_labels):
        word_loss = self.word_loss(word_logits, word_labels.float())
        syllable_loss = self.syllable_loss(syllable_logits, syllable_labels.float())

        loss = word_loss + self.syllable_loss_lambda * syllable_loss

        if self.align_loss_lambda > 0:
            mask = word_labels.float()
            align_loss = (self.align_loss(word_logits, syllable_logits) * mask).sum() / mask.sum()
            loss = loss + self.align_loss_lambda * align_loss

            return loss, word_loss, syllable_loss, align_loss

        return loss, word_loss, syllable_loss, 0.0

# class PositionalEncoding(nn.Module):
#     def __init__(self, embed_dim, seq_len):
#         super().__init__()
#         self.pe = torch.tensor([
#             [pos / (10000.0 ** (i // 2 * 2.0 / embed_dim)) for i in range(embed_dim)]
#             for pos in range(seq_len)])
#         self.pe[:, 0::2] = torch.sin(self.pe[:, 0::2])
#         self.pe[:, 1::2] = torch.cos(self.pe[:, 1::2])
#         self.pe = torch.unsqueeze(self.pe, dim=1)
#         self.pe = nn.Parameter(self.pe, requires_grad=False)
#
#     def forward(self, x):
#         return x + self.pe
