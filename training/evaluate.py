from pathlib import Path

import torch
from torch import nn

from app.ml.model import CLASS_NAMES, create_model
from training.prepare_data import create_dataloaders


MODEL_PATH = Path("models/efficientnet_b0_best.pt")


def print_metrics(confusion_matrix: torch.Tensor) -> None:
    total = confusion_matrix.sum().item()
    correct = confusion_matrix.diag().sum().item()

    print(f"Test accuracy: {correct / total:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix)

    for index, class_name in enumerate(CLASS_NAMES):
        true_positive = confusion_matrix[index, index].item()
        false_positive = confusion_matrix[:, index].sum().item() - true_positive
        false_negative = confusion_matrix[index, :].sum().item() - true_positive

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0

        print(
            f"{class_name}: "
            f"precision={precision:.4f}, "
            f"recall={recall:.4f}"
        )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, class_names = create_dataloaders()

    if tuple(class_names) != CLASS_NAMES:
        raise RuntimeError(
            f"Unexpected class order: {class_names}. Expected: {CLASS_NAMES}."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model = create_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    confusion_matrix = torch.zeros(
        len(CLASS_NAMES),
        len(CLASS_NAMES),
        dtype=torch.int64,
    )

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images).argmax(dim=1)

            for actual, predicted in zip(labels.cpu(), predictions.cpu()):
                confusion_matrix[actual, predicted] += 1

    print_metrics(confusion_matrix)


if __name__ == "__main__":
    main()