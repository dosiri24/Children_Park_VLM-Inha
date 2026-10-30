"""Read prompt files as-is and version them by content hash."""

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Prompts:
    """Hold raw prompt texts and their combined content hash."""

    system: str
    views: dict[str, str]
    personas: dict[str, str]
    request: str
    version: str


def load_prompts(directory: Path, directions: list[str], persona_ids: list[str]) -> Prompts:
    """Read system, view, persona, and request files without modification."""

    def read(name: str) -> str:
        """Read one prompt file relative to the prompt directory."""

        return (directory / name).read_text(encoding="utf-8")

    files = {
        "system.md": read("system.md"),
        "request.md": read("request.md"),
        **{f"views/{direction}.md": read(f"views/{direction}.md") for direction in directions},
        **{f"personas/{pid}.md": read(f"personas/{pid}.md") for pid in persona_ids},
    }
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(f"{name}\0{files[name]}\0".encode("utf-8"))
    return Prompts(
        system=files["system.md"],
        views={direction: files[f"views/{direction}.md"] for direction in directions},
        personas={pid: files[f"personas/{pid}.md"] for pid in persona_ids},
        request=files["request.md"],
        version=digest.hexdigest(),
    )
