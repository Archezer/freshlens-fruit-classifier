from pathlib import Path
import shutil

import torch

from app.ml.model import CLASS_NAMES, create_model
from training.prepare_data import create_dataloaders


MODEL_PATH = Path("models/efficientnet_b0_best.pt")
OUTPUT_DIR = Path("reports/false_negatives")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, _ = create_dataloaders()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model = create_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rotten_index = CLASS_NAMES.index("rotten")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_index = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            predictions = model(images.to(device)).argmax(dim=1)

            for label, prediction in zip(labels, predictions.cpu()):
                source_path = test_loader.dataset.samples[image_index][0]

                if label.item() == rotten_index and prediction.item() != rotten_index:
                    shutil.copy2(source_path, OUTPUT_DIR / Path(source_path).name)

                image_index += 1

    print(f"False negatives saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()