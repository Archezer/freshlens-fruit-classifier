from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from app.ml.model import CLASS_NAMES, create_model
from app.ml.preprocessing import create_eval_transform
from training.evaluate import print_metrics


MODEL_PATH = Path("models/efficientnet_b0_final.pt")
EXTERNAL_TEST_DIR = Path("data/external_test")
BATCH_SIZE = 16


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageFolder(
        EXTERNAL_TEST_DIR,
        transform=create_eval_transform(),
    )

    if tuple(dataset.classes) != CLASS_NAMES:
        raise RuntimeError(
            f"Unexpected class order: {dataset.classes}. "
            f"Expected: {CLASS_NAMES}."
        )

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model = create_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    confusion_matrix = torch.zeros(2, 2, dtype=torch.int64)

    with torch.inference_mode():
        for images, labels in data_loader:
            predictions = model(images.to(device)).argmax(dim=1)

            for actual, predicted in zip(labels, predictions.cpu()):
                confusion_matrix[actual, predicted] += 1

    print_metrics(confusion_matrix)


if __name__ == "__main__":
    main()
