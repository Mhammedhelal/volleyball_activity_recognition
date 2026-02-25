import torch
import torch.nn as nn
from ..utils.subgroups import make_subgroup_indices



class SubGroupPooler(nn.Module):
    """
    Intra-group pooling + frame-level concatenation (Section 3.3, Eq. 9–13).

    At each timestep t:
        G_{t,m} = MaxPool / AvgPool ( P_{t,k}  for k ∈ subgroup_m )   (Eq. 12)
        Z_t     = G_{t,1} ⊕ … ⊕ G_{t,d}                               (Eq. 13)

    Stacking over all T gives the sequence Z fed to LSTM2.

    Input  : P   [N, T, D+H]          fused person embeddings
    Output : Z   [1, T, z_dim]        frame descriptors ready for LSTM2
                                       z_dim = (D+H) * n_subgroups
    """

    def __init__(self, pool: str = "max"):
        super().__init__()
        assert pool in ("max", "avg"), f"pool must be 'max' or 'avg', got '{pool}'"
        self.pool = pool

    def forward(
        self,
        P:                torch.Tensor,
        subgroup_indices: list[list[int]],
    ) -> torch.Tensor:
        """
        P                : [N, T, D+H]
        subgroup_indices : list of d index lists (one list per sub-group)

        Returns
          Z : [1, T, z_dim]
        """
        N, T, embed_dim = P.shape
        Z_sequence = []

        for t in range(T):
            P_t = P[:, t, :]          # [N, D+H] — all persons at timestep t

            group_vecs = []
            for idx in subgroup_indices:
                sub = P_t[idx]        # [N_sub, D+H]
                if self.pool == "max":
                    g = sub.max(dim=0).values   # [D+H]
                else:
                    g = sub.mean(dim=0)         # [D+H]
                group_vecs.append(g)

            Z_t = torch.cat(group_vecs, dim=-1)   # [z_dim]
            Z_sequence.append(Z_t)

        Z = torch.stack(Z_sequence, dim=0).unsqueeze(0)   # [1, T, z_dim]
        return Z
