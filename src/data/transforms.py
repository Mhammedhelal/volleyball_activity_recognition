from torchvision import transforms

# ImageNet defaults — hardcoded to avoid import-time config dependency.
# These match default.yaml exactly and should not change between experiments.
IMAGE_SIZE    = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# -- Training transforms ------------------------------------------------------
# NOTE: RandomHorizontalFlip is intentionally omitted — flipping reverses
# left/right team assignments, corrupting group-activity labels.
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

# -- Validation / Test transforms ---------------------------------------------
eval_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])