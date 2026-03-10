from pathlib import Path
import sys

import torch
import torch.nn as nn

from src.config import Config

# Resolve config path relative to project root
config_path = Path(__file__).resolve().parent.parent.parent / 'configs' / 'default.yaml'
cfg = Config.from_yaml(config_path)
LSTM_HIDDEN_G = cfg.group_lstm.hidden_dim

GROUP_ACTIVITIES = cfg.labels.group_activities
#[
#     "l_set", "l_spike", "l_pass", "l_winpoint",
#     "r_set", "r_spike", "r_pass", "r_winpoint",
# ]  # 8 classes


class FrameDescriptor(nn.Module):
    """
    Stage 2 — Group-level temporal model (Section 3.2).

    Receives the full T-length sequence of frame descriptors Z_1 … Z_T
    and models how the group activity evolves over time.

        h^group_t = LSTM2(Z_t, h^group_{t-1})
        ŷ         = Softmax(W * h^group_T + b)

    Input  : Z   [1, T, z_dim]    output of SubGroupPooler
    Output : group_logits  [8]    one score per group activity class
    """

    def __init__(
        self,
        z_dim:         int,
        lstm_hidden:   int = LSTM_HIDDEN_G,
        group_classes: int = len(GROUP_ACTIVITIES),   # 8
        n_layers:      int = 1,
    ):
        super().__init__()
        self.group_lstm = nn.LSTM(
            input_size  = z_dim,
            hidden_size = lstm_hidden,
            num_layers  = n_layers,
            batch_first = True,
        )
        self.group_fc    = nn.Linear(lstm_hidden, group_classes)
        self.lstm_hidden = lstm_hidden
        self.n_layers    = n_layers

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z : [1, T, z_dim]

        Returns
          group_logits : [8]
        """
        lstm_out, _  = self.group_lstm(Z)          # [1, T, lstm_hidden]
        h_group      = lstm_out[0, -1, :]          # [lstm_hidden]  last hidden state
        group_logits = self.group_fc(h_group)      # [8]
        return group_logits
