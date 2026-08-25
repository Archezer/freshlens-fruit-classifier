import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
)
from torchvision.transforms import functional as transform_functional

from app.ml.model import CLASS_NAMES
from app.ml.predict import DEVICE, load_model
from app.ml.preprocessing import (
    IMAGE_CROP,
    IMAGE_RESIZE,
    create_eval_transform,
)


def generate_gradcam(
    image: Image.Image,
    class_name: str | None = None,
) -> Image.Image:
    model = load_model()
    rgb_image = image.convert("RGB")

    input_tensor = create_eval_transform()(
        rgb_image
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        predicted_index = model(input_tensor).argmax(dim=1).item()

    target_index = (
        CLASS_NAMES.index(class_name)
        if class_name is not None
        else predicted_index
    )

    target_layers = [model.features[-1]]

    with GradCAM(
        model=model,
        target_layers=target_layers,
    ) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(target_index)],
        )[0]

    visual_image = transform_functional.center_crop(
        transform_functional.resize(
            rgb_image,
            IMAGE_RESIZE,
        ),
        [IMAGE_CROP, IMAGE_CROP],
    )

    visual_array = np.asarray(
        visual_image,
        dtype=np.float32,
    ) / 255.0

    heatmap = show_cam_on_image(
        visual_array,
        grayscale_cam,
        use_rgb=True,
    )

    return Image.fromarray(heatmap)