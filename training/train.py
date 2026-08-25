from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from app.ml.model import CLASS_NAMES, create_model
from training.prepare_data import create_dataloaders


EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MODEL_PATH = Path("models/efficientnet_b0_best.pt")
ROTTEN_CLASS_WEIGHT = 1.75


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_function: nn.Module,
) -> tuple[float, float, float]:
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    confusion_matrix = torch.zeros(
        len(CLASS_NAMES),
        len(CLASS_NAMES),
        dtype=torch.int64,
    )

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_function(logits, labels)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct_predictions += (
                predictions == labels
            ).sum().item()
            total_images += images.size(0)

            for actual, predicted in zip(
                labels.cpu(),
                predictions.cpu(),
            ):
                confusion_matrix[actual, predicted] += 1

    rotten_index = CLASS_NAMES.index("rotten")

    true_positive = confusion_matrix[
        rotten_index,
        rotten_index,
    ].item()
    false_positive = (
        confusion_matrix[:, rotten_index].sum().item()
        - true_positive
    )
    false_negative = (
        confusion_matrix[rotten_index, :].sum().item()
        - true_positive
    )

    rotten_f1 = (
        2 * true_positive
        / (2 * true_positive + false_positive + false_negative)
        if 2 * true_positive + false_positive + false_negative
        else 0.0
    )

    return (
        total_loss / total_images,
        correct_predictions / total_images,
        rotten_f1,
    )

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    train_loader, validation_loader, _, class_names = create_dataloaders()

    if tuple(class_names) != CLASS_NAMES:
        raise RuntimeError(
            f"Unexpected class order: {class_names}. Expected: {CLASS_NAMES}."
        )

    model = create_model().to(device)
    class_weights = torch.tensor(
        [1.0, ROTTEN_CLASS_WEIGHT],
        device=device,
    )

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    best_validation_rotten_f1 = 0.0
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = loss_function(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        validation_loss, validation_accuracy, validation_rotten_f1 = evaluate(
            model,
            validation_loader,
            device,
            loss_function,
        )

        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"validation_loss={validation_loss:.4f} | "
            f"validation_accuracy={validation_accuracy:.4f} | "
            f"validation_rotten_f1={validation_rotten_f1:.4f}"
        )

        if validation_rotten_f1 > best_validation_rotten_f1:
            best_validation_rotten_f1 = validation_rotten_f1

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "validation_accuracy": validation_accuracy,
                    "validation_rotten_f1": validation_rotten_f1,
                },
                MODEL_PATH,
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "validation_accuracy": validation_accuracy,
                },
                MODEL_PATH,
            )

    print(
        f"Best validation rotten F1: "
        f"{best_validation_rotten_f1:.4f}"
    )


if __name__ == "__main__":
    main()