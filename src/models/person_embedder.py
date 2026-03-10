from pathlib import Path
import sys
from typing import Callable
import torch
import torch.nn as nn
from torchvision import models

PERSON_ACTIONS = [
    "waiting", "setting", "digging", "falling",
    "spiking", "blocking", "jumping", "moving", "standing"
]  # 9 classes


def build_alexnet_fc7(freeze: bool = True) -> tuple[nn.Module, int]:
    """
    Returns (model, output_dim=4096).
    Frozen AlexNet up to and including fc7 (classifier[5] = second ReLU).

    AlexNet classifier layout:
        [0] Dropout
        [1] Linear(9216, 4096)   ← fc6
        [2] ReLU
        [3] Dropout
        [4] Linear(4096, 4096)   ← fc7
        [5] ReLU                 ← stop here
        [6] Linear(4096, 1000)   ← ImageNet head, discarded
    """
    alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    feature_extractor = nn.Sequential(
        alexnet.features,
        alexnet.avgpool,
        nn.Flatten(),
        *list(alexnet.classifier.children())[:6],
    )
    if freeze:
        for p in feature_extractor.parameters():
            p.requires_grad = False

    feature_extractor.eval()

    return feature_extractor, 4096

def build_resnet50(freeze: bool = True) -> tuple[nn.Module, int]:
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    backbone = nn.Sequential(
        *list(resnet.children())[:-1],
        nn.Flatten()
    )

    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False

    backbone.eval()

    return backbone, 2048

def build_mobilenet_v3_large(freeze: bool = True) -> tuple[nn.Module, int]:

    mobilenet = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.DEFAULT
    )

    backbone = nn.Sequential(
        mobilenet.features,
        mobilenet.avgpool,
        nn.Flatten()
    )

    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False

    backbone.eval()

    return backbone, 960  


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
