import base64
from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.ml.gradcam import generate_gradcam
from app.ml.predict import predict_image
from app.schemas.prediction import FruitQuality, PredictionResponse

router = APIRouter()
MAXIMUM_IMAGE_PIXELS = 12_000_000


@router.get('/health')
def health_check() -> dict[str, str]:
    return {'status': 'ok'}


@router.post(
    '/predict',
    response_model=PredictionResponse,
)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail='Only image files are supported'
        )

    file_bytes = await file.read()

    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail='Image must not exceed 10 MB'
        )

    try:
        with Image.open(BytesIO(file_bytes)) as source_image:
            if source_image.width * source_image.height > MAXIMUM_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail='Image resolution must not exceed 12 megapixels',
                )
            image = source_image.convert('RGB')
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='The uploaded file is not a valid image.',
        ) from error

    probabilities = predict_image(image)
    predicted_class = max(probabilities, key=probabilities.get)
    heatmap = generate_gradcam(image, predicted_class)

    heatmap_buffer = BytesIO()
    heatmap.save(heatmap_buffer, format='JPEG', quality=90)
    heatmap_base64 = base64.b64encode(
        heatmap_buffer.getvalue()
    ).decode('ascii')

    return PredictionResponse(
        predicted_class=FruitQuality(predicted_class),
        confidence=probabilities[predicted_class],
        probabilities=probabilities,
        heatmap_data_url=f'data:image/jpeg;base64,{heatmap_base64}',
    )
