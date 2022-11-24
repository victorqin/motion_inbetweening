import torch
import torch.nn as nn

from motion_inbetween.model import transformer


class ContextTransformer(nn.Module):
    def __init__(self, config):
        super(ContextTransformer, self).__init__()
        self.config = config

        self.d_mask = config["d_mask"]
        self.constrained_slices = [
            slice(*i) for i in config["constrained_slices"]
        ]

        self.dropout = config["dropout"]
        self.pre_lnorm = config["pre_lnorm"]
        self.n_layer = config["n_layer"]

        self.encoder = nn.Sequential(
            nn.Linear(self.config["d_encoder_in"], self.config["d_encoder_h"]),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.config["d_encoder_h"], self.config["d_model"]),
            nn.PReLU(),
            nn.Dropout(self.dropout)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.config["d_model"], self.config["d_decoder_h"]),
            nn.PReLU(),
            nn.Linear(self.config["d_decoder_h"], self.config["d_out"])
        )

        self.rel_pos_layer = nn.Sequential(
            nn.Linear(1, self.config["d_head"]),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.config["d_head"], self.config["d_head"]),
            nn.Dropout(self.dropout)
        )

        self.keyframe_pos_layer = nn.Sequential(
            nn.Linear(2, self.config["d_model"]),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.config["d_model"], self.config["d_model"]),
            nn.Dropout(self.dropout)
        )

        self.layer_norm = nn.LayerNorm(self.config["d_model"])
        self.att_layers = nn.ModuleList()
        self.pff_layers = nn.ModuleList()

        for i in range(self.n_layer):
            self.att_layers.append(
                transformer.RelMultiHeadedAttention(
                    self.config["n_head"], self.config["d_model"],
                    self.config["d_head"], dropout=self.config["dropout"],
                    pre_lnorm=self.config["pre_lnorm"],
                    bias=self.config["atten_bias"]
                )
            )

            self.pff_layers.append(
                transformer.PositionwiseFeedForward(
                    self.config["d_model"], self.config["d_pff_inner"],
                    dropout=self.config["dropout"],
                    pre_lnorm=self.config["pre_lnorm"]
                )
            )

    def get_rel_pos_emb(self, window_len, dtype, device):
        pos_idx = torch.arange(-window_len + 1, window_len,
                               dtype=dtype, device=device)
        pos_idx = pos_idx[None, :, None]        # (1, seq, 1)
        rel_pos_emb = self.rel_pos_layer(pos_idx)
        return rel_pos_emb

    def forward(self, x, keyframe_pos, mask=None):
        x = self.encoder(x)

        x = x + self.keyframe_pos_layer(keyframe_pos)

        rel_pos_emb = self.get_rel_pos_emb(x.shape[-2], x.dtype, x.device)

        for i in range(self.n_layer):
            x = self.att_layers[i](x, rel_pos_emb, mask=mask)
            x = self.pff_layers[i](x)
        if self.pre_lnorm:
            x = self.layer_norm(x)
        x = self.decoder(x)

        return x


class DetailTransformer(nn.Module):
    def __init__(self, config):
        super(DetailTransformer, self).__init__()
        self.config = config

        self.d_mask = config["d_mask"]
        self.constrained_slices = [
            slice(*i) for i in config["constrained_slices"]
        ]

        self.dropout = config["dropout"]
        self.pre_lnorm = config["pre_lnorm"]
        self.n_layer = config["n_layer"]

        self.encoder = nn.Sequential(
            nn.Linear(self.config["d_encoder_in"], self.config["d_encoder_h"]),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.config["d_encoder_h"], self.config["d_model"]),
            nn.PReLU(),
            nn.Dropout(self.dropout)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.config["d_model"], self.config["d_decoder_h"]),
            nn.PReLU(),
            nn.Linear(self.config["d_decoder_h"], self.config["d_out"])
        )

        self.rel_pos_layer = nn.Sequential(
            nn.Linear(1, self.config["d_head"]),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.config["d_head"], self.config["d_head"]),
            nn.Dropout(self.dropout)
        )

        self.layer_norm = nn.LayerNorm(self.config["d_model"])
        self.att_layers = nn.ModuleList()
        self.pff_layers = nn.ModuleList()

        for i in range(self.n_layer):
            self.att_layers.append(
                transformer.RelMultiHeadedAttention(
                    self.config["n_head"], self.config["d_model"],
                    self.config["d_head"], dropout=self.config["dropout"],
                    pre_lnorm=self.config["pre_lnorm"],
                    bias=self.config["atten_bias"]
                )
            )

            self.pff_layers.append(
                transformer.PositionwiseFeedForward(
                    self.config["d_model"], self.config["d_pff_inner"],
                    dropout=self.config["dropout"],
                    pre_lnorm=self.config["pre_lnorm"]
                )
            )

    def get_rel_pos_emb(self, window_len, dtype, device):
        pos_idx = torch.arange(-window_len + 1, window_len,
                               dtype=dtype, device=device)
        pos_idx = pos_idx[None, :, None]        # (1, seq, 1)
        rel_pos_emb = self.rel_pos_layer(pos_idx)
        return rel_pos_emb

    def forward(self, x, mask=None):
        x = self.encoder(x)
        rel_pos_emb = self.get_rel_pos_emb(x.shape[-2], x.dtype, x.device)

        for i in range(self.n_layer):
            x = self.att_layers[i](x, rel_pos_emb, mask=mask)
            x = self.pff_layers[i](x)
        if self.pre_lnorm:
            x = self.layer_norm(x)
        x = self.decoder(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config

        self.layers = nn.Sequential(
            nn.Conv1d(config["d_in"], config["d_conv1"],
                      kernel_size=config["kernel_size"],
                      stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(config["d_conv1"], config["d_conv2"],
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(config["d_conv2"], 1, kernel_size=1,
                      stride=1, padding=0),
        )

    def forward(self, data):
        data = data.transpose(-1, -2)     # (batch, dim, seq)
        return self.layers(data)          # (batch, 1, seq)
