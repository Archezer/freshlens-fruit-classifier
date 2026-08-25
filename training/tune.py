import json
from pathlib import Path

import optuna
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from app.ml.model import CLASS_NAMES, create_model
from training.prepare_data import create_dataloaders


EPOCHS_PER_TRIAL = 5
TRIALS = 20
SEED = 42
MIN_GOOD_RECALL = 0.70
RESULT_PATH = Path("models/optuna_best_params.json")


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    confusion_matrix = torch.zeros(2, 2, dtype=torch.int64)

    with torch.inference_mode():
        for images, labels in data_loader:
            predictions = model(images.to(device)).argmax(dim=1)

            for actual, predicted in zip(labels, predictions.cpu()):
                confusion_matrix[actual, predicted] += 1

    good_index = CLASS_NAMES.index("good")
    rotten_index = CLASS_NAMES.index("rotten")

    rotten_tp = confusion_matrix[rotten_index, rotten_index].item()
    rotten_fp = (
        confusion_matrix[:, rotten_index].sum().item() - rotten_tp
    )
    rotten_fn = (
        confusion_matrix[rotten_index, :].sum().item() - rotten_tp
    )

    rotten_f1 = (
        2 * rotten_tp
        / (2 * rotten_tp + rotten_fp + rotten_fn)
    )

    good_tp = confusion_matrix[good_index, good_index].item()
    good_fn = (
        confusion_matrix[good_index, :].sum().item() - good_tp
    )
    good_recall = good_tp / (good_tp + good_fn)

    return rotten_f1, good_recall


def objective(trial: optuna.Trial) -> float:
    device = torch.device("cuda")

    batch_size = trial.suggest_categorical(
        "batch_size",
        [16, 32, 64],
    )
    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        3e-3,
        log=True,
    )
    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        1e-3,
        log=True,
    )
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    rotten_weight = trial.suggest_float(
        "rotten_weight",
        1.0,
        2.5,
    )

    train_loader, validation_loader, _, _ = create_dataloaders(
        batch_size=batch_size,
    )

    model = create_model(dropout=dropout).to(device)

    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [1.0, rotten_weight],
            device=device,
        )
    )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS_PER_TRIAL,
    )
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                loss = loss_function(model(images), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        rotten_f1, good_recall = evaluate(
            model,
            validation_loader,
            device,
        )

        score = rotten_f1
        if good_recall < MIN_GOOD_RECALL:
            score -= (MIN_GOOD_RECALL - good_recall) * 2

        trial.report(score, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("rotten_f1", rotten_f1)
    trial.set_user_attr("good_recall", good_recall)

    return score


def main() -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
        ),
    )

    study.optimize(objective, n_trials=TRIALS)

    print("\nBest score:", study.best_value)
    print("Best parameters:", study.best_params)
    print("Best rotten F1:", study.best_trial.user_attrs["rotten_f1"])
    print("Best good recall:", study.best_trial.user_attrs["good_recall"])

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(study.best_params, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()