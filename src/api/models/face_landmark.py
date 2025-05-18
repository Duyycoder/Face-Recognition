from __future__ import annotations

from typing import Any
from typing import List

from common.base import BaseModel


class APIInput(BaseModel):
    img: List[List[List[int]]]
    bbox: List[Any]


class APIOutput(BaseModel):
    pred: Any
