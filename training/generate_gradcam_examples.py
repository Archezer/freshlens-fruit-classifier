from pathlib import Path

from PIL import Image

from app.ml.gradcam import generate_gradcam


SOURCE_DIR = Path("data/external_test")
OUTPUT_DIR = Path("reports/gradcam_examples")
CLASSES = ("good", "rotten")
IMAGES_PER_CLASS = 3


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for class_name in CLASSES:
        images = sorted(
            (SOURCE_DIR / class_name).glob("*.jpg")
        )[:IMAGES_PER_CLASS]

        for image_path in images:
            heatmap = generate_gradcam(
                Image.open(image_path)
            )

            output_path = (
                OUTPUT_DIR
                / f"{class_name}_{image_path.stem}_cam.jpg"
            )

            heatmap.save(output_path)
            print(output_path)


if __name__ == "__main__":
    main()