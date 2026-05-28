from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class InputType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TEXT_AND_IMAGE = "text_and_image"


class RunRequest(BaseModel):
    content: str | list[dict[str, Any]]
    extra_body: dict[str, Any] = Field(default_factory=dict)


class RunResponse(BaseModel):
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None


class InfoResponse(BaseModel):
    input_type: InputType
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]