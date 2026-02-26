from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Training transforms ───────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # FIX: RandomHorizontalFlip is NOT safe for volleyball.
    # Flipping a crop mirrors the player's body which is fine, but it also
    # silently mirrors the spatial meaning of "left" vs "right" actions and
    # subgroup assignments (left_team / right_team). Because group labels
    # encode side (left_spike, right_pass, etc.), a flipped crop would be
    # paired with the wrong group label. Removed.
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── Validation / Test transforms ─────────────────────────────────────────────
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])