from pathlib import Path
import random
import shutil


RAW_DIR = Path("data/raw")
DATA_DIR = Path("data")

CLASSES = ("good", "rotten")
MENDELEY_IMAGES_PER_CLASS = 200
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def ensure_empty_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"{path} is not empty. Refusing to overwrite split data."
        )

    path.mkdir(parents=True, exist_ok=True)


def copy_images(images: list[Path], destination: Path) -> None:
    for image in images:
        shutil.copy2(image, destination / image.name)


def split_class(class_name: str) -> tuple[int, int, int]:
    images = sorted((RAW_DIR / class_name).glob("*"))

    if len(images) <= MENDELEY_IMAGES_PER_CLASS:
        raise RuntimeError(f"Not enough images for class: {class_name}")

    source_groups = {
        "mendeley": images[:MENDELEY_IMAGES_PER_CLASS],
        "fruitvision": images[MENDELEY_IMAGES_PER_CLASS:],
    }

    split_images: dict[str, list[Path]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for source_name, source_images in source_groups.items():
        shuffled_images = source_images.copy()

        random.Random(
            f"{SEED}:{class_name}:{source_name}"
        ).shuffle(shuffled_images)

        test_count = round(len(shuffled_images) * TEST_RATIO)
        validation_count = round(
            len(shuffled_images) * VALIDATION_RATIO
        )

        split_images["test"].extend(
            shuffled_images[:test_count]
        )
        split_images["validation"].extend(
            shuffled_images[test_count:test_count + validation_count]
        )
        split_images["train"].extend(
            shuffled_images[test_count + validation_count:]
        )

    destinations = {
        split_name: DATA_DIR / split_name / class_name
        for split_name in split_images
    }

    for destination in destinations.values():
        ensure_empty_directory(destination)

    for split_name, images_to_copy in split_images.items():
        copy_images(images_to_copy, destinations[split_name])

    return (
        len(split_images["train"]),
        len(split_images["validation"]),
        len(split_images["test"]),
    )


def main() -> None:
    for class_name in CLASSES:
        train_count, validation_count, test_count = split_class(
            class_name
        )

        print(
            f"{class_name}: "
            f"train={train_count}, "
            f"validation={validation_count}, "
            f"test={test_count}"
        )


if __name__ == "__main__":
    main()