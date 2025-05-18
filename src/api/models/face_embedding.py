from __future__ import annotations

from typing import List

from common.base import BaseModel


class APIInput(BaseModel):
    image: List[List[List[int]]]
    landmarks: List[List[float]]


class APIOutput(BaseModel):
    face_embedding: list[float]
