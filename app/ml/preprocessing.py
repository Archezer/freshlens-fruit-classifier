from collections.abc import Callable

from torchvision import transforms


IMAGE_RESIZE = 256
IMAGE_CROP = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_RESIZE),
            transforms.RandomCrop(IMAGE_CROP),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

def create_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_RESIZE),
            transforms.CenterCrop(IMAGE_CROP),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )