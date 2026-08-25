import os
from dataclasses import dataclass, field


def read_cors_origins() -> list[str]:
    raw_origins = os.getenv(
        'CORS_ALLOWED_ORIGINS',
        'http://localhost:5173,http://127.0.0.1:5173',
    )
    return [
        origin.strip()
        for origin in raw_origins.split(',')
        if origin.strip()
    ]


@dataclass(frozen=True)
class Settings:
    app_name: str = 'Freshlens API'
    app_version: str = '0.1.0'
    debug: bool = True
    cors_allowed_origins: list[str] = field(
        default_factory=read_cors_origins
    )

settings = Settings()
