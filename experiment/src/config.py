"""Load the experiment configuration."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"


class Strict(BaseModel):
    """Reject unknown configuration keys."""

    model_config = ConfigDict(extra="forbid")


class PersonaConfig(Strict):
    """Describe one persona condition."""

    id: str
    persona_type: str
    child_age_group: str | None
    child_gender: Literal["male", "female"] | None


class ImageConfig(Strict):
    """Describe in-memory image encoding and requested media resolution."""

    resolution: Literal["low", "medium", "high", "ultra_high"]
    max_side: int = Field(gt=0)
    jpeg_quality: int = Field(ge=1, le=100)


class PricingConfig(Strict):
    """Store token prices in US dollars per million tokens."""

    input: float
    cached_input: float
    output: float


class Config(Strict):
    """Store the complete experiment configuration."""

    model_id: str
    thinking_level: str
    temperature: float
    request_timeout_seconds: int = Field(gt=0)
    cache_ttl_seconds: int = Field(gt=0)
    prompt_dir: Path
    photo_dir: Path
    directions: list[str]
    results_dir: Path
    log_dir: Path
    image: ImageConfig
    repeats: int = Field(gt=0)
    parallel_parks: int = Field(gt=0)
    personas: list[PersonaConfig]
    pricing_usd_per_million_tokens: PricingConfig


def load_config(path: Path) -> Config:
    """Load a YAML config and resolve its paths relative to the config file."""

    config = Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    base = path.resolve().parent
    return config.model_copy(
        update={
            name: (base / getattr(config, name)).resolve()
            for name in ("prompt_dir", "photo_dir", "results_dir", "log_dir")
        }
    )
