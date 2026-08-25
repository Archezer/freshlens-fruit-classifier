from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from app.ml.preprocessing import (
    create_eval_transform,
    create_train_transform,
)


DATA_ROOT = Path("data")
BATCH_SIZE = 32
NUM_WORKERS = 0


def create_dataloaders(
    batch_size: int = BATCH_SIZE,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    list[str]
]:
    train_dataset = ImageFolder(
        DATA_ROOT / 'train',
        transform=create_train_transform(),
    )
    validation_dataset = ImageFolder(
        DATA_ROOT / 'validation',
        transform=create_eval_transform(),
    )
    test_dataset = ImageFolder(
        DATA_ROOT / 'test',
        transform=create_eval_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        train_dataset.classes,
    )
