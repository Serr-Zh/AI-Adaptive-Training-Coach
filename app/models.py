from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class InputType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TEXT_AND_IMAGE = "text_and_image"


class ContentPartText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ContentPartImage(BaseModel):
    type: Literal["image"] = "image"
    image: str = Field(description="Base64-encoded image data (PNG, JPEG, etc.)")


ContentPart = Annotated[
    Union[ContentPartText, ContentPartImage],
    Field(discriminator="type"),
]


class RunRequest(BaseModel):
    content: str | list[ContentPart] = Field(
        description=(
            "Main input: plain text string or list of typed content parts. "
            "Use a string for text-only input. "
            "Use a list for image or multimodal input."
        ),
    )
    extra_body: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional parameters (temperature, max_tokens, etc.)",
    )


class RunResponse(BaseModel):
    status: Literal["success", "error"]
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class InfoResponse(BaseModel):
    input_type: InputType
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
