"""Cache park images explicitly and evaluate them per persona with Gemini."""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field


# Gemini writes JSON keys in schema order, so this order is the order the model judges elements in.
ELEMENT_NAMES = {
    "safety": "안전성",
    "openness": "개방성",
    "aesthetics": "심미성",
    "diversity": "다양성",
    "maintenance": "관리성",
    "playfulness": "놀이성",
    "sociality": "사회성",
    "general_impression": "종합성",
}


class ElementEvaluation(BaseModel):
    """Store one evidence-first element evaluation."""

    model_config = ConfigDict(extra="forbid")

    reason: str
    score: int = Field(ge=1, le=4)


class Evaluation(BaseModel):
    """Store all eight element evaluations."""

    model_config = ConfigDict(extra="forbid")

    safety: ElementEvaluation
    openness: ElementEvaluation
    aesthetics: ElementEvaluation
    diversity: ElementEvaluation
    maintenance: ElementEvaluation
    playfulness: ElementEvaluation
    sociality: ElementEvaluation
    general_impression: ElementEvaluation


@dataclass(frozen=True)
class EncodedImage:
    """Store one transmitted JPEG with its source and pixel size."""

    direction: str
    file: str
    data: bytes
    width: int
    height: int


@dataclass(frozen=True)
class CallResult:
    """Return the raw response, parsed evaluation, token usage, and any error."""

    raw: dict[str, Any]
    parsed: dict[str, Any] | None
    usage: dict[str, int]
    error: str | None


def encode_image(path: Path, direction: str, max_side: int, jpeg_quality: int) -> EncodedImage:
    """Downscale an EXIF-oriented RGB copy and encode it as JPEG bytes."""

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    return EncodedImage(direction, path.name, buffer.getvalue(), image.width, image.height)


def response_schema() -> dict[str, Any]:
    """Build the JSON schema with reason before score for every element."""

    element = {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "score": {"type": "integer", "minimum": 1, "maximum": 4},
        },
        "required": ["reason", "score"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {name: element for name in ELEMENT_NAMES},
        "required": list(ELEMENT_NAMES),
        "additionalProperties": False,
    }


def park_contents(
    view_texts: dict[str, str], images: list[EncodedImage], resolution: str
) -> list[types.Content]:
    """Build the cached user turn of photo tags and park images."""

    level = types.PartMediaResolutionLevel(f"MEDIA_RESOLUTION_{resolution.upper()}")
    parts = []
    for image in images:
        parts.append(types.Part(text=view_texts[image.direction]))
        parts.append(
            types.Part.from_bytes(
                data=image.data,
                mime_type="image/jpeg",
                media_resolution=types.PartMediaResolution(level=level),
            )
        )
    return [types.Content(role="user", parts=parts)]


class GeminiClient:
    """Create park caches and send persona evaluations through generate_content."""

    def __init__(self, api_key: str, timeout_seconds: int) -> None:
        """Create the SDK client with a per-request timeout."""

        self.client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=timeout_seconds * 1000)
        )

    def create_cache(
        self,
        model_id: str,
        system_text: str,
        contents: list[types.Content],
        ttl_seconds: int,
        display_name: str,
    ) -> types.CachedContent:
        """Cache the shared system text and one park's images."""

        return self.client.caches.create(
            model=model_id,
            config=types.CreateCachedContentConfig(
                system_instruction=system_text,
                contents=contents,
                ttl=f"{ttl_seconds}s",
                display_name=display_name,
            ),
        )

    def delete_cache(self, name: str) -> None:
        """Delete a park cache to stop storage billing."""

        self.client.caches.delete(name=name)

    def evaluate(
        self,
        model_id: str,
        cache_name: str,
        persona_text: str,
        request_text: str,
        thinking_level: str,
        temperature: float,
    ) -> CallResult:
        """Evaluate one cached park with persona and request text as the final user turn."""

        # The no-persona condition has an empty persona file, so only the request part is sent.
        parts = [types.Part(text=text) for text in (persona_text, request_text) if text]
        response = self.client.models.generate_content(
            model=model_id,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                cached_content=cache_name,
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                temperature=temperature,
                response_mime_type="application/json",
                response_json_schema=response_schema(),
            ),
        )
        raw = response.model_dump(mode="json", exclude_none=True)
        usage = response.usage_metadata
        tokens = {
            "input_tokens": (usage.prompt_token_count or 0) if usage else 0,
            "cached_tokens": (usage.cached_content_token_count or 0) if usage else 0,
            "output_tokens": (usage.candidates_token_count or 0) if usage else 0,
            "thought_tokens": (usage.thoughts_token_count or 0) if usage else 0,
        }
        if not response.text:
            finish = response.candidates[0].finish_reason if response.candidates else None
            return CallResult(raw, None, tokens, f"empty response: finish_reason={finish}")
        try:
            parsed = Evaluation.model_validate_json(response.text).model_dump(mode="json")
        except ValueError as exception:
            return CallResult(raw, None, tokens, f"structured output parse failed: {exception}")
        return CallResult(raw, parsed, tokens, None)
