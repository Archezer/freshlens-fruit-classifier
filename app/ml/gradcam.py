import numpy as np
import torch
from PIL import Image
from torch.nn import functional as functional
from torchvision.transforms import functional as transform_functional

from app.ml.model import CLASS_NAMES
from app.ml.predict import DEVICE, load_model
from app.ml.preprocessing import IMAGE_CROP, IMAGE_RESIZE, create_eval_transform


def create_heatmap_colors(values: np.ndarray) -> np.ndarray:
    scaled_values = 4.0 * values
    red = np.clip(1.5 - np.abs(scaled_values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(scaled_values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(scaled_values - 1.0), 0.0, 1.0)
    return np.stack((red, green, blue), axis=-1)


def generate_gradcam(
    image: Image.Image,
    class_name: str | None = None,
) -> Image.Image:
    model = load_model()
    rgb_image = image.convert('RGB')
    input_tensor = create_eval_transform()(rgb_image).unsqueeze(0).to(DEVICE)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    target_layer = model.features[-1]

    forward_handle = target_layer.register_forward_hook(
        lambda _module, _inputs, output: activations.append(output.detach()),
    )
    backward_handle = target_layer.register_full_backward_hook(
        lambda _module, _grad_inputs, grad_outputs: gradients.append(
            grad_outputs[0].detach(),
        ),
    )

    try:
        with torch.enable_grad():
            logits = model(input_tensor)
            predicted_index = logits.argmax(dim=1).item()
            target_index = (
                CLASS_NAMES.index(class_name)
                if class_name is not None
                else predicted_index
            )

            model.zero_grad(set_to_none=True)
            logits[0, target_index].backward()

            channel_weights = gradients[-1].mean(dim=(2, 3), keepdim=True)
            class_activation = torch.relu(
                (channel_weights * activations[-1]).sum(dim=1, keepdim=True),
            )
            class_activation = functional.interpolate(
                class_activation,
                size=(IMAGE_CROP, IMAGE_CROP),
                mode='bilinear',
                align_corners=False,
            )[0, 0]
    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)

    heatmap_values = class_activation.detach().cpu().numpy()
    heatmap_values -= heatmap_values.min()
    heatmap_maximum = heatmap_values.max()
    if heatmap_maximum > 0:
        heatmap_values /= heatmap_maximum

    visual_image = transform_functional.center_crop(
        transform_functional.resize(rgb_image, IMAGE_RESIZE),
        [IMAGE_CROP, IMAGE_CROP],
    )
    visual_values = np.asarray(visual_image, dtype=np.float32) / 255.0
    overlay = 0.55 * visual_values + 0.45 * create_heatmap_colors(heatmap_values)

    return Image.fromarray(np.uint8(np.clip(overlay, 0.0, 1.0) * 255))
