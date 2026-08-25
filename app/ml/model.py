from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)

CLASS_NAMES = ("good", "rotten")
NUM_CLASSES = len(CLASS_NAMES)


def create_model() -> nn.Module:
    model = efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT,
    )

    input_features = next(
        layer.in_features
        for layer in reversed(list(model.classifier.modules()))
        if isinstance(layer, nn.Linear)
    )

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(input_features, NUM_CLASSES),
    )

    return model