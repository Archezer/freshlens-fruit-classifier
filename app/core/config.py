from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = 'Freshlens API'
    app_version: str = '0.1.0'
    debug: bool = True

settings = Settings()