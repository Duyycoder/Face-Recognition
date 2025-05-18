from __future__ import annotations

from dotenv import find_dotenv
from dotenv import load_dotenv
from pydantic import HttpUrl

from pydantic_settings import BaseSettings

from .models import DetectorSettings
from .models import EmbeddingSettings
from .models import FaceAlignSettings
# test in local
load_dotenv(find_dotenv('.env'), override=True)


class Settings(BaseSettings):
    host_embedding: HttpUrl
    host_detector: HttpUrl
    host_landmark: HttpUrl

    embedding: EmbeddingSettings
    face_align: FaceAlignSettings
    detector: DetectorSettings

    class Config:
        env_nested_delimiter = '__'
