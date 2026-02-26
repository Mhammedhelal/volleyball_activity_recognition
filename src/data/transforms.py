from torchvision import transforms

# AlexNet / ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ------------------------------------------------
# Training transforms
# ------------------------------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),          # AlexNet input
    transforms.RandomHorizontalFlip(p=0.5), # safe for volleyball
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ------------------------------------------------
# Validation / Test transforms
# ------------------------------------------------
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])