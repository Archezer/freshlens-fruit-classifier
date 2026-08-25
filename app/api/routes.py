from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas.prediction import PredictionResponse

router = APIRouter()


@router.get('/health')
def health_check() -> dict[str, str]:
    return {'status': 'ok'}


@router.post(
    '/predict',
    response_model=PredictionResponse,
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Model inference is not implemented yet.",
    )