from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

from app.ml.model import CLASS_NAMES, create_model
from app.ml.preprocessing import create_eval_transform


MODEL_PATH = Path("models/efficientnet_b0_final.pt")
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


@lru_cache
def load_model() -> torch.nn.Module:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = create_model().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def predict_image(image: Image.Image) -> dict[str, float]:
    model = load_model()

    tensor = create_eval_transform()(
        image.convert('RGB')
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        probabilities = torch.softmax(
            model(tensor),
            dim=1
        )[0]

    return {
        class_name: probabilities[index].item()
        for index, class_name in enumerate(CLASS_NAMES)
    }
