from typing import Callable

import torch
import torch.nn as nn

from src.models.person_embedder import PersonEmbedder, PERSON_ACTIONS, build_alexnet_fc7
from src.models.subgroup_pooler import SubGroupPooler
from src.utils.subgroups import make_subgroup_indices
from src.models.frame_descriptor import FrameDescriptor, GROUP_ACTIVITIES


class HierarchicalGroupActivityModel(nn.Module):
    """
    Full two-stage hierarchical model for group activity recognition.
    Ibrahim et al., "Hierarchical Deep Temporal Models for Group Activity
    Recognition", IEEE TPAMI.

    Pipeline (Section 3, Eq. 7–13):

      Stage 1 — Person level  (PersonEmbedder)
        x_{t,k}  = CNN(B_{t,k})
        h_{t,k}  = LSTM1(x_{t,k}, h_{t-1,k})
        P_{t,k}  = x_{t,k} ⊕ h_{t,k}              at every t

      Stage 2a — Sub-group pooling  (SubGroupPooler)
        G_{t,m}  = Pool( P_{t,k}  for k ∈ subgroup_m )
        Z_t      = G_{t,1} ⊕ … ⊕ G_{t,d}

      Stage 2b — Group temporal model  (FrameDescriptor)
        h^group  = LSTM2(Z_1, …, Z_T)
        ŷ        = Softmax(W * h^group_T + b)

    Input  : x  [N, T, C, H, W]
    Output :
        group_logits   [8]     main supervision — team activity label
        person_logits  [N, 9]  auxiliary supervision — individual action labels
    """

    def __init__(
        self,
        feature_extractor: Callable[[], tuple[nn.Module, int]] = build_alexnet_fc7,    # CNN backbone function
        lstm_hidden_p:   int = 3000,    # LSTM1 hidden size
        lstm_hidden_g:   int = 2000,    # LSTM2 hidden size
        person_classes:  int = len(PERSON_ACTIONS),    # 9
        group_classes:   int = len(GROUP_ACTIVITIES),  # 8
        n_subgroups:     int = 2,      # 1=all | 2=left/right | 4=back/front
        pool:            str = "max",  # "max" or "avg"
        n_layers_p:      int = 1,      # LSTM1 depth
        n_layers_g:      int = 1,      # LSTM2 depth
    ):
        super().__init__()
        self.n_subgroups = n_subgroups

        self.person_embedder  = PersonEmbedder(
            feature_extractor = feature_extractor,
            lstm_hidden     = lstm_hidden_p,
            person_classes  = person_classes,
            n_layers        = n_layers_p,
        )
        embed_dim = self.person_embedder.cnn_dim + lstm_hidden_p   # D+H per person
        z_dim     = embed_dim * n_subgroups           # Z_t dimension

        self.subgroup_pooler  = SubGroupPooler(pool=pool)
        self.frame_descriptor = FrameDescriptor(
            z_dim         = z_dim,
            lstm_hidden   = lstm_hidden_g,
            group_classes = group_classes,
            n_layers      = n_layers_g,
        )
    def forward(
        self,
        x:                torch.Tensor | None = None,
        P:                torch.Tensor | None = None,
        subgroup_indices: list[list[int]] | None = None,
    ):
        """
        Provide exactly one of:
            x : [N, T, C, H, W]  raw player crops   — runs person_embedder (CNN+LSTM1)
            P : [N, T, D+H]      cached Stage-1 output — skips CNN+LSTM1 entirely

        Returns
        group_logits   : [8]
        person_logits  : [N, 9]
        """
        assert (x is None) != (P is None), "Pass exactly one of x or P"

        if x is not None:
            N = x.shape[0]
            person_logits, P = self.person_embedder(x)         # [N,9], [N,T,D+H]
        else:
            N = P.shape[0]
            # person_embedder is frozen whenever P is cached, so person_fc is the
            # only op needed to recover person_logits — no CNN/LSTM1 forward pass.
            person_logits = self.person_embedder.person_fc(P[:, -1, :])

        if subgroup_indices is None:
            subgroup_indices = make_subgroup_indices(N, self.n_subgroups)

        Z = self.subgroup_pooler(P, subgroup_indices)           # [1, T, z_dim]
        group_logits = self.frame_descriptor(Z)                 # [8]

        return group_logits, person_logits
