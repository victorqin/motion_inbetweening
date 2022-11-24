import torch
import torch.nn as nn


class TargetNoiseEmbedding(nn.Module):
    def __init__(self, dim, sigma):
        super(TargetNoiseEmbedding, self).__init__()
        self.dim = dim
        self.sigma = sigma
        self.device = None

    def forward(self, tta):
        """
        Args:
            tta (int): number of frames away from target frame
        Returns:
            tensor: target noise tensor, shape: (dim, )
        """
        if tta < 5:
            lambda_target = 0
        elif tta < 30:
            lambda_target = (tta - 5.0) / 25
        else:
            lambda_target = 1

        mean = torch.zeros(self.dim, device=self.device)
        std = torch.full((self.dim,), self.sigma, device=self.device)

        return lambda_target * torch.normal(mean, std)

    def to(self, device):
        self.device = device
        return super(TargetNoiseEmbedding, self).to(device)


class TimeToArrivalEmbedding(nn.Module):
    def __init__(self, dim, context_len, max_trans, base=10000):
        super(TimeToArrivalEmbedding, self).__init__()
        self.dim = dim
        self.base = base
        self.context_len = context_len
        self.max_trans = max_trans

        inv_freq = 1 / (base ** (torch.arange(0.0, dim, 2.0) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tta):
        """
        Args:
            tta (int): number of frames away from target frame
        Returns:
            tensor: time-to-arrival embedding, shape: (dim,)
        """
        tta = min(tta, self.context_len + self.max_trans - 5)

        inv_term = tta * self.inv_freq

        # interleave sin and cos values
        pe = torch.stack([inv_term.sin(), inv_term.cos()], dim=-1)
        pe = pe.flatten(start_dim=-2)

        # ":self.dim" on last dimesion makes sure the dimension of
        # positional encoding is as required when self.dim is
        # an odd number.
        return pe[..., :self.dim]


class RmiMotionGenerator(nn.Module):
    def __init__(self, config):
        super(RmiMotionGenerator, self).__init__()
        self.config = config

        self.state_encoder = nn.Sequential(
            nn.Linear(config["d_state_in"], config["d_encoder_h"]),
            nn.PReLU(),
            nn.Linear(config["d_encoder_h"], config["d_encoder_out"]),
            nn.PReLU()
        )

        self.offset_encoder = nn.Sequential(
            nn.Linear(config["d_offset_in"], config["d_encoder_h"]),
            nn.PReLU(),
            nn.Linear(config["d_encoder_h"], config["d_encoder_out"]),
            nn.PReLU()
        )

        self.target_encoder = nn.Sequential(
            nn.Linear(config["d_target_in"], config["d_encoder_h"]),
            nn.PReLU(),
            nn.Linear(config["d_encoder_h"], config["d_encoder_out"]),
            nn.PReLU()
        )

        self.lstm = nn.LSTM(config["d_encoder_out"] * 3, config["d_lstm_h"],
                            config["lstm_layer"])

        self.decoder = nn.Sequential(
            nn.Linear(config["d_lstm_h"], config["d_decoder_h1"]),
            nn.PReLU(),
            nn.Linear(config["d_decoder_h1"], config["d_decoder_h2"]),
            nn.PReLU(),
            nn.Linear(config["d_decoder_h2"], config["d_decoder_out"]),
        )

    def forward(self, state, offset, target, hidden, indices,
                ztta, ztarget=None):
        """
        Args:
            state (tensor): current state, shape: (batch, dim)
            offset (tensor): current offset, shape: (batch, dim)
            target (tensor): current target, shape: (batch, dim)
            hidden (tuple of tensor): (h, c), shape of h and c:
                (lstm_layer, batch, d_lstm_h).
                For initialization, set hidden to None.
            indices (dict): config which defines the meaning of input's
                dimensions
            ztta (tensor): shape: (d_encoder_out, )
            ztarget (tensor or None, optional): shape: (d_encoder_out*2, ).
                When None, no noise is added. Defaults to None.

        Returns:
            tuple of tensor: (next_state, next_hidden)
                next_state: shape: (batch, dim)
                next_hidden: hidden state of LSTM, (h, c), shape same as input
        """
        state_emb = self.state_encoder(state)
        offset_emb = self.offset_encoder(offset)
        target_emb = self.target_encoder(target)

        # add ztta
        state_emb = state_emb + ztta
        offset_emb = offset_emb + ztta
        target_emb = target_emb + ztta

        ot_emb = torch.cat((offset_emb, target_emb), dim=-1)

        # add ztarget
        if ztarget is not None:
            ot_emb = ot_emb + ztarget

        lstm_in = torch.cat([state_emb, ot_emb], dim=-1)
        lstm_in = lstm_in.unsqueeze(0)          # (1, batch, dim)

        # lstm_out: (1, batch, d_lstm_h)
        # hidden: (h, c)
        lstm_out, next_hidden = self.lstm(lstm_in, hidden)

        c_slice = slice(indices["c_start_idx"], indices["c_end_idx"])
        state_out = self.decoder(lstm_out).squeeze(0)
        next_state = state + state_out
        next_state[..., c_slice] = torch.sigmoid(state_out[..., c_slice])

        return next_state, next_hidden


class RmiMotionDiscriminator(nn.Module):
    def __init__(self, config):
        super(RmiMotionDiscriminator, self).__init__()
        self.config = config
        self.layers = nn.Sequential(
            nn.Conv1d(config["d_in"], config["d_hidden1"],
                      kernel_size=config["window_len"],
                      stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(config["d_hidden1"], config["d_hidden2"], kernel_size=1,
                      stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(config["d_hidden2"], 1, kernel_size=1,
                      stride=1, padding=0)
        )

    def forward(self, state):
        """
        Arguments:
            data: (batch, frame, dim)

        Returns:
            scores of each batch. 1 means good, 0 means bad.
            (batch, 1)
        """
        state = state.transpose(1, 2)
        return self.layers(state).mean(dim=-1)
