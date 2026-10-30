"""Run the park x persona x repeat evaluation matrix with resumable JSONL storage."""

import argparse
import csv
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from config import CONFIG_PATH, Config, PersonaConfig, load_config
from gemini import (
    ELEMENT_NAMES,
    CallResult,
    EncodedImage,
    GeminiClient,
    encode_image,
    park_contents,
)
from prompts import load_prompts


PROJECT_DIR = Path(__file__).resolve().parents[2]
CALLS_FILE = "calls.jsonl"
SCORES_FILE = "scores.csv"
KEY_FIELDS = ["park_id", "persona_id", "repeat"]
SCORE_FIELDS = [
    "park_id",
    "park_name",
    "persona_id",
    "persona_type",
    "child_age_group",
    "child_gender",
    "repeat",
    "element",
    "element_ko",
    "score",
    "reason",
    "model_id",
    "prompt_version",
    "timestamp",
]


def configure_logging(log_dir: Path) -> logging.Logger:
    """Log to a timestamped file and the console."""

    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in (
        logging.FileHandler(log_dir / f"run_{stamp}.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def find_parks(photo_dir: Path) -> dict[str, dict[str, Any]]:
    """Group photos named <park_id>_<park_name>_<direction>.JPG by park."""

    parks: dict[str, dict[str, Any]] = defaultdict(lambda: {"photos": {}})
    for path in sorted(photo_dir.glob("*.JPG")):
        park_id, park_name, direction = path.stem.split("_")
        parks[park_id]["name"] = park_name
        parks[park_id]["photos"][direction] = path
    return dict(sorted(parks.items()))


def read_calls(path: Path) -> list[dict[str, Any]]:
    """Read every stored call record from the JSONL file."""

    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def completed_keys(
    calls: list[dict[str, Any]], config: Config, prompt_version: str
) -> set[tuple[Any, ...]]:
    """Return successful call keys, refusing to resume across model, generation, or prompt changes."""

    for call in calls:
        if (
            call["model_id"] != config.model_id
            or call["thinking_level"] != config.thinking_level
            or call["temperature"] != config.temperature
            or call["prompt_version"] != prompt_version
        ):
            raise ValueError(
                "calls.jsonl contains a different model_id, thinking_level, temperature, or "
                "prompt_version; use a new results_dir for changed settings"
            )
    return {tuple(call[field] for field in KEY_FIELDS) for call in calls if call["error"] is None}


def append_call(path: Path, record: dict[str, Any]) -> None:
    """Append one call record as a single JSONL line and flush it to disk."""

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def export_scores(calls_path: Path, scores_path: Path) -> int:
    """Rebuild the long-format score table from successful call records."""

    rows = [
        {
            **{field: call[field] for field in SCORE_FIELDS[:7]},
            "element": element,
            "element_ko": element_ko,
            "score": call["parsed"][element]["score"],
            "reason": call["parsed"][element]["reason"],
            "model_id": call["model_id"],
            "prompt_version": call["prompt_version"],
            "timestamp": call["timestamp"],
        }
        for call in read_calls(calls_path)
        if call["error"] is None
        for element, element_ko in ELEMENT_NAMES.items()
    ]
    with scores_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def call_record(
    config: Config,
    prompt_version: str,
    park_id: str,
    park_name: str,
    persona: PersonaConfig,
    repeat: int,
    timestamp: str,
    elapsed_seconds: float,
    cache_name: str,
    persona_text: str,
    images: list[EncodedImage],
    result: CallResult | None,
    error: str | None,
) -> dict[str, Any]:
    """Build one raw call record with identifiers, parameters, usage, and response."""

    return {
        "park_id": park_id,
        "park_name": park_name,
        "persona_id": persona.id,
        "persona_type": persona.persona_type,
        "child_age_group": persona.child_age_group,
        "child_gender": persona.child_gender,
        "repeat": repeat,
        "model_id": config.model_id,
        "thinking_level": config.thinking_level,
        "temperature": config.temperature,
        "prompt_version": prompt_version,
        "timestamp": timestamp,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "request": {
            # System text is identified by prompt_version; only the varying persona text is stored.
            "cache_name": cache_name,
            "persona_text": persona_text,
            "timeout_seconds": config.request_timeout_seconds,
            "image": config.image.model_dump(),
            "photos": [
                {
                    "direction": image.direction,
                    "file": image.file,
                    "width": image.width,
                    "height": image.height,
                    "jpeg_bytes": len(image.data),
                }
                for image in images
            ],
        },
        "usage": result.usage if result else None,
        "response": result.raw if result else None,
        "parsed": result.parsed if result else None,
        "error": error or (result.error if result else None),
    }


def run(config_path: Path) -> None:
    """Run pending calls for all parks, skipping calls already stored as successful."""

    config = load_config(config_path)
    logger = configure_logging(config.log_dir)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    calls_path = config.results_dir / CALLS_FILE
    scores_path = config.results_dir / SCORES_FILE
    try:
        prompts = load_prompts(
            config.prompt_dir, config.directions, [persona.id for persona in config.personas]
        )
        parks = find_parks(config.photo_dir)
        done = completed_keys(read_calls(calls_path), config, prompts.version)
    except Exception:
        logger.exception("Setup failed: config=%s", config_path)
        raise

    total = len(parks) * len(config.personas) * config.repeats
    logger.info(
        "Run start: config=%s model=%s thinking=%s temperature=%s prompt_version=%s parks=%d "
        "personas=%d repeats=%d total_calls=%d parallel_parks=%d already_done=%d results=%s",
        config_path.resolve(), config.model_id, config.thinking_level, config.temperature,
        prompts.version, len(parks), len(config.personas), config.repeats, total,
        config.parallel_parks, len(done), calls_path,
    )

    client = GeminiClient(load_api_key(PROJECT_DIR / ".env"), config.request_timeout_seconds)
    pricing = config.pricing_usd_per_million_tokens
    totals = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "thought_tokens": 0}
    counter = {"index": len(done)}
    # Parks run concurrently, so file appends and shared counters go through one lock.
    lock = threading.Lock()

    def run_park(park_id: str, park: dict[str, Any]) -> None:
        """Cache one park, run its pending persona calls in order, and delete the cache."""

        pending = [
            (persona, repeat)
            for persona in config.personas
            for repeat in range(1, config.repeats + 1)
            if (park_id, persona.id, repeat) not in done
        ]
        if not pending:
            logger.info("Skip park %s %s: all calls done", park_id, park["name"])
            return

        images = [
            encode_image(
                park["photos"][direction], direction,
                config.image.max_side, config.image.jpeg_quality,
            )
            for direction in config.directions
        ]
        try:
            cache = client.create_cache(
                config.model_id, prompts.system,
                park_contents(prompts.views, images, config.image.resolution),
                config.cache_ttl_seconds, park_id,
            )
        except Exception:
            logger.exception("Cache creation failed: park=%s %s", park_id, park["name"])
            raise
        logger.info(
            "Cache created: park=%s %s name=%s cached_tokens=%s pending_calls=%d",
            park_id, park["name"], cache.name,
            cache.usage_metadata.total_token_count if cache.usage_metadata else None,
            len(pending),
        )
        try:
            for persona, repeat in pending:
                persona_text = prompts.personas[persona.id]
                timestamp = datetime.now(timezone.utc).isoformat()
                started = time.monotonic()
                result = None
                error = None
                try:
                    result = client.evaluate(
                        config.model_id, cache.name, persona_text, prompts.request,
                        config.thinking_level, config.temperature,
                    )
                except Exception as exception:
                    error = f"{type(exception).__name__}: {exception}"
                    logger.exception(
                        "Call raised: park=%s persona=%s repeat=%d", park_id, persona.id, repeat
                    )
                record = call_record(
                    config, prompts.version, park_id, park["name"], persona, repeat,
                    timestamp, time.monotonic() - started, cache.name, persona_text,
                    images, result, error,
                )
                with lock:
                    append_call(calls_path, record)
                    counter["index"] += 1
                    if result:
                        for name in totals:
                            totals[name] += result.usage[name]
                    # prompt_token_count includes cached tokens; thinking tokens bill at the output rate.
                    cost = (
                        (totals["input_tokens"] - totals["cached_tokens"]) * pricing.input
                        + totals["cached_tokens"] * pricing.cached_input
                        + (totals["output_tokens"] + totals["thought_tokens"]) * pricing.output
                    ) / 1_000_000
                    logger.log(
                        logging.ERROR if record["error"] else logging.INFO,
                        "Call %d/%d %s: park=%s persona=%s repeat=%d elapsed=%.1fs usage=%s "
                        "error=%s session_cost_usd=%.4f",
                        counter["index"], total, "FAILED" if record["error"] else "done",
                        park_id, persona.id, repeat, record["elapsed_seconds"], record["usage"],
                        record["error"], cost,
                    )
        finally:
            client.delete_cache(cache.name)
            logger.info("Cache deleted: park=%s name=%s", park_id, cache.name)

    try:
        with ThreadPoolExecutor(max_workers=config.parallel_parks) as pool:
            futures = [pool.submit(run_park, park_id, park) for park_id, park in parks.items()]
            for future in as_completed(futures):
                future.result()
    finally:
        rows = export_scores(calls_path, scores_path) if calls_path.exists() else 0
        calls = read_calls(calls_path)
        failed = sum(call["error"] is not None for call in calls)
        succeeded = completed_keys(calls, config, prompts.version)
        logger.info(
            "Run end: stored_successful_calls=%d/%d score_rows=%d failed_attempts=%d "
            "(failed keys are retried on the next run) session_tokens=%s scores=%s",
            len(succeeded), total, rows, failed, totals, scores_path,
        )


def load_api_key(env_path: Path) -> str:
    """Load the Gemini API key from the project .env file."""

    load_dotenv(dotenv_path=env_path)
    key = os.getenv("gemini_api_key")
    if not key:
        raise ValueError("gemini_api_key is missing from the project .env file")
    return key


def main() -> None:
    """Parse command-line options and run the experiment."""

    parser = argparse.ArgumentParser(description="Run the four-view park evaluation.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
