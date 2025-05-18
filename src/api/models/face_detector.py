"""Face detector API format input """
from __future__ import annotations

from common.base import BaseModel


class APIOutput(BaseModel):
    bboxes: list
    landmarks: list
