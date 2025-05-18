from __future__ import annotations

from pydantic import BaseModel

class CustomBaseModel(BaseModel):
    class Config:
        """Configuration of the Pydantic Object"""
        arbitrary_types_allowed = True