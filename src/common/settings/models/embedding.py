from __future__ import annotations

from common.base import BaseModel


class EmbeddingSettings(BaseModel):
    model_path: str
    input_mean_available: float
    input_std_available: float
    input_mean_unavailable: float
    input_std_unavailable: float
