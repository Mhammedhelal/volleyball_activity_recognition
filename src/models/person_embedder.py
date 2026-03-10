from pathlib import Path
import sys
from typing import Callable
import torch
import torch.nn as nn
from torchvision import models
from src.models.cnn_backbones import build_alexnet_fc7

from src.config import Config
# Resolve config path relative to project root
config_path = Path(__file__).resolve().parent.parent.parent / 'configs' / 'default.yaml'
cfg = Config.from_yaml(config_path)

PERSON_ACTIONS = cfg.labels.person_actions





class PersonEmbedder(nn.Module):
    """
    Stage 1 — Person-level spatial + temporal encoding.

    For every person crop at every timestep:
        x_{t,k} = CNN(B_{t,k})              spatial feature   [D]
        h_{t,k} = LSTM1(x_{t,k}, h_{t-1,k}) temporal feature  [H]
        P_{t,k} = x_{t,k} ⊕ h_{t,k}        fused embedding   [D+H]

    The shared CNN has no temporal awareness (processes one crop at a time).
    The shared LSTM1 processes one person sequence at a time.
    Fusion happens at EVERY timestep so that downstream pooling
    can produce a Z_t per frame, giving LSTM2 a proper T-length sequence.

    Input  : x  [N, T, C, H, W]
    Output :
        person_logits  [N, 9]       supervised by individual action labels
        P              [N, T, D+H]  fused embedding at every timestep
    """

    def __init__(
        self,
        feature_extractor: Callable[[], tuple[nn.Module, int]] = build_alexnet_fc7,          
        lstm_hidden:     int = 512,
        person_classes:  int = len(PERSON_ACTIONS),
        n_layers:        int = 1,
    ):
        super().__init__()
        self.cnn, cnn_out_dim = feature_extractor()


        self.lstm = nn.LSTM(
            input_size  = cnn_out_dim,
            hidden_size = lstm_hidden,
            num_layers  = n_layers,
            batch_first = True,
        )
        self.person_fc = nn.Linear(cnn_out_dim + lstm_hidden, person_classes)

        self.cnn_dim     = cnn_out_dim
        self.hidden_size = lstm_hidden
        self.n_layers    = n_layers

    def forward(self, x: torch.Tensor):
        """
        x : [N, T, C, H, W]
        """
        N, T, C, H, W = x.shape

        # CNN — one crop at a time, no temporal awareness
        cnn_out = self.cnn(x.view(N * T, C, H, W))   # [N*T, D]
        cnn_out = cnn_out.view(N, T, self.cnn_dim)    # [N,   T, D]

        # LSTM1 — one person sequence at a time (shared weights across persons)
        lstm_out, _ = self.lstm(cnn_out)   # [N, T, H]

        # P_{t,k} = x_{t,k} ⊕ h_{t,k}  at every t  (Eq. 7)
        P = torch.cat([cnn_out, lstm_out], dim=-1)    # [N, T, D+H]

        # Person-action head — supervised at last timestep
        person_logits = self.person_fc(P[:, -1, :])   # [N, 9]

        return person_logits, P
