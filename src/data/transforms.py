from pathlib import Path
import sys
from torchvision import transforms

from src.config import Config

# Resolve config path relative to project root
config_path = Path(__file__).resolve().parent.parent.parent / 'configs' / 'default.yaml'
cfg = Config.from_yaml(config_path)
IMAGE_SIZE =  cfg.dataset.image_size

IMAGENET_MEAN = cfg.dataset.mean
IMAGENET_STD  = cfg.dataset.std

# -- Training transforms ------------------------------------------------------
train_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# -- Validation / Test transforms ------------------------------------------------------
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])