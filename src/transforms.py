from torchvision import transforms

from .config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def get_train_transform(augmentation=True):
    transform_list = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]

    if augmentation:
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    return transforms.Compose(transform_list)


def get_validation_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def get_robust_train_transform():
    """
    Enhanced augmentation pipeline designed to improve model robustness
    against real-world image perturbations (shadows, blur, occlusion, perspective).
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.15,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.2),
            value="random",
        ),
    ])