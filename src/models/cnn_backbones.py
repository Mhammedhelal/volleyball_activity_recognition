from torchvision import models
import torch.nn as nn

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