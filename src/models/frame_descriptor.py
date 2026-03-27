import torch
import torch.nn as nn

# Constants that must match default.yaml — kept here to avoid import-time
# config loading (which breaks if the config path moves).
_DEFAULT_LSTM_HIDDEN_G = 2000
_DEFAULT_GROUP_CLASSES  = 8

# Imported for use by hierarchical_model.py / models/__init__.py
from src.data.labels import GROUP_ACTIVITIES


class FrameDescriptor(nn.Module):
    """
    Stage 2b — Group-level temporal model (Section 3.2).

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
        lstm_hidden:   int = _DEFAULT_LSTM_HIDDEN_G,
        group_classes: int = _DEFAULT_GROUP_CLASSES,
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
          group_logits : [group_classes]
        """
        lstm_out, _  = self.group_lstm(Z)       # [1, T, lstm_hidden]
        h_group      = lstm_out[0, -1, :]       # [lstm_hidden]  last hidden state
        group_logits = self.group_fc(h_group)   # [group_classes]
        return group_logits