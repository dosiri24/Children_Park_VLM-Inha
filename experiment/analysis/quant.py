"""Compute descriptive statistics, paired t-tests, repeat consistency, and direction citations."""

import argparse
import logging
import re
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd
import yaml
from scipy import stats


ANALYSIS_DIR = Path(__file__).resolve().parent
DIRECTION_KO = {"북": "N", "동": "E", "남": "S", "서": "W"}
MARKER = re.compile(r"\(([북동남서]쪽(?:·[북동남서]쪽)*)에서 촬영\)")
CELL = ["park_id", "persona_id", "element"]


def configure_logging() -> logging.Logger:
    """Log to a timestamped file and the console."""

    log_dir = ANALYSIS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger = logging.getLogger("analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.FileHandler(log_dir / f"quant_{stamp}.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_scores(scores_path: Path, repeats: int) -> pd.DataFrame:
    """Load scores and require exactly one row per park, persona, element, and repeat."""

    scores = pd.read_csv(scores_path)
    counts = scores.groupby(CELL)["repeat"].agg(["count", "nunique"])
    bad = counts[(counts["count"] != repeats) | (counts["nunique"] != repeats)]
    if not bad.empty:
        raise ValueError(f"cells without exactly {repeats} repeats: {bad.index.tolist()}")
    return scores


def ordered(frame: pd.DataFrame, personas: list[str], elements: list[str]) -> pd.DataFrame:
    """Sort rows by configured persona order and element order."""

    keys = {"persona_id": personas, "persona_a": personas, "persona_b": personas, "element": elements}
    columns = [name for name in keys if name in frame.columns]
    return frame.sort_values(
        columns,
        key=lambda column: column.map({value: i for i, value in enumerate(keys[column.name])}),
        kind="stable",
    )


def score_distribution(scores: pd.DataFrame) -> pd.DataFrame:
    """Count raw scores 1-4 per persona and element, with an all-element row."""

    overall = scores.assign(element="all", element_ko="전체")
    table = (
        pd.concat([scores, overall])
        .groupby(["persona_id", "element", "element_ko"])["score"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=[1, 2, 3, 4], fill_value=0)
    )
    table.columns = [f"score_{value}" for value in table.columns]
    table["n"] = table.sum(axis=1)
    return table.reset_index()


def cell_stats(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize each park x persona x element cell over repeats."""

    grouped = scores.groupby(["park_id", "park_name", "persona_id", "element", "element_ko"])["score"]
    cells = grouped.agg(mean="mean", sd="std", min="min", max="max").reset_index()
    cells["range"] = cells["max"] - cells["min"]
    return cells.drop(columns=["min", "max"])


def persona_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """Describe the park means per element and persona."""

    return (
        cells.groupby(["element", "element_ko", "persona_id"])["mean"]
        .agg(n_parks="count", mean="mean", sd="std", min="min", max="max")
        .reset_index()
    )


def paired_tests(cells: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Compare persona_a with persona_b per element using paired t-tests on park means."""

    wide = cells.pivot_table(index=["element", "element_ko", "park_id"], columns="persona_id", values="mean")
    rows = []
    for (element, element_ko), parks in wide.groupby(level=["element", "element_ko"]):
        for persona_a, persona_b in pairs:
            # Park means are multiples of 1/repeats, so rounding removes float noise before sign counts.
            diff = (parks[persona_a] - parks[persona_b]).round(9)
            ttest = stats.ttest_rel(parks[persona_a], parks[persona_b])
            rows.append(
                {
                    "element": element,
                    "element_ko": element_ko,
                    "persona_a": persona_a,
                    "persona_b": persona_b,
                    "n_parks": len(diff),
                    "mean_a": parks[persona_a].mean(),
                    "mean_b": parks[persona_b].mean(),
                    "mean_diff": diff.mean(),
                    "sd_diff": diff.std(),
                    "t": ttest.statistic,
                    "df": len(diff) - 1,
                    "p": ttest.pvalue,
                    "parks_higher": int((diff > 0).sum()),
                    "parks_equal": int((diff == 0).sum()),
                    "parks_lower": int((diff < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def discrimination(cells: pd.DataFrame, elements: list[str]) -> pd.DataFrame:
    """Compare the spread of park means with the spread of persona means per element."""

    overall = cells.groupby(["park_id", "persona_id"])["mean"].mean().reset_index().assign(element="all")
    rows = []
    for element in [*elements, "all"]:
        subset = overall if element == "all" else cells[cells["element"] == element]
        park_means = subset.groupby("park_id")["mean"].mean()
        persona_means = subset.groupby("persona_id")["mean"].mean()
        rows.append(
            {
                "element": element,
                "park_mean_min": park_means.min(),
                "park_mean_max": park_means.max(),
                "park_range": park_means.max() - park_means.min(),
                "park_sd": park_means.std(),
                "persona_mean_min": persona_means.min(),
                "persona_mean_max": persona_means.max(),
                "persona_range": persona_means.max() - persona_means.min(),
            }
        )
    return pd.DataFrame(rows)


def repeat_consistency(cells: pd.DataFrame, personas: list[str], elements: list[str]) -> pd.DataFrame:
    """Aggregate repeat standard deviations and ranges by persona, element, and overall."""

    def summarize(frame: pd.DataFrame, group_type: str, group: str) -> dict:
        """Summarize one group of cells."""

        return {
            "group_type": group_type,
            "group": group,
            "n_cells": len(frame),
            "mean_sd": frame["sd"].mean(),
            "max_sd": frame["sd"].max(),
            "range_0": int((frame["range"] == 0).sum()),
            "range_1": int((frame["range"] == 1).sum()),
            "range_2plus": int((frame["range"] >= 2).sum()),
        }

    rows = [summarize(cells, "overall", "all")]
    rows += [summarize(cells[cells["persona_id"] == p], "persona", p) for p in personas]
    rows += [summarize(cells[cells["element"] == e], "element", e) for e in elements]
    rows += [
        summarize(cells[(cells["persona_id"] == p) & (cells["element"] == e)], "persona_element", f"{p}:{e}")
        for p in personas
        for e in elements
    ]
    return pd.DataFrame(rows)


def direction_citations(scores: pd.DataFrame, personas: list[str], directions: list[str]) -> pd.DataFrame:
    """Count cited photo directions and directions per marker by persona and overall."""

    records = []
    for persona, reason in zip(scores["persona_id"], scores["reason"]):
        for marker in MARKER.findall(reason):
            cited = [DIRECTION_KO[part[0]] for part in marker.split("·")]
            records.append({"persona_id": persona, "directions": cited})
    markers = pd.DataFrame(records)

    def summarize(frame: pd.DataFrame, group: str, reasons: int) -> dict:
        """Summarize the markers of one persona group."""

        row = {"persona_id": group, "reasons": reasons, "markers": len(frame)}
        mentions = sum(len(cited) for cited in frame["directions"])
        for direction in directions:
            count = sum(direction in cited for cited in frame["directions"])
            row[f"cite_{direction}"] = count
            row[f"share_{direction}"] = count / mentions
        for size in range(1, len(directions) + 1):
            row[f"markers_{size}dir"] = int((frame["directions"].map(len) == size).sum())
        return row

    reasons = scores.groupby("persona_id").size()
    rows = [summarize(markers[markers["persona_id"] == p], p, int(reasons[p])) for p in personas]
    rows.append(summarize(markers, "all", len(scores)))
    return pd.DataFrame(rows)


def run(config_path: Path) -> None:
    """Run all quantitative analyses on the configured results."""

    logger = configure_logging()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results_dir = (config_path.resolve().parent / config["results_dir"]).resolve()
    personas = [persona["id"] for persona in config["personas"]]
    baseline = next(p["id"] for p in config["personas"] if p["persona_type"] == "none")
    children = [persona for persona in personas if persona != baseline]
    out_dir = ANALYSIS_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = load_scores(results_dir / "scores.csv", config["repeats"])
    elements = list(dict.fromkeys(scores["element"]))
    logger.info(
        "loaded %d rows from %s: %d parks, personas=%s, elements=%s, repeats=%d, prompt_versions=%s",
        len(scores), results_dir / "scores.csv", scores["park_id"].nunique(), personas, elements,
        config["repeats"], scores["prompt_version"].unique().tolist(),
    )

    cells = cell_stats(scores)
    tables = {
        "score_distribution.csv": ordered(score_distribution(scores), personas, [*elements, "all"]),
        "persona_summary.csv": ordered(persona_summary(cells), personas, elements),
        "paired_vs_baseline.csv": ordered(
            paired_tests(cells, [(child, baseline) for child in children]), personas, elements
        ),
        "paired_child_pairs.csv": ordered(
            paired_tests(cells, list(combinations(children, 2))), personas, elements
        ),
        "discrimination.csv": discrimination(cells, elements),
        "repeat_consistency.csv": repeat_consistency(cells, personas, elements),
        "direction_citations.csv": direction_citations(scores, personas, config["directions"]),
    }
    for name, table in tables.items():
        table.to_csv(out_dir / name, index=False, encoding="utf-8-sig", float_format="%.4f")
        logger.info("wrote %s (%d rows)", out_dir / name, len(table))

    for name in ("paired_vs_baseline.csv", "paired_child_pairs.csv"):
        tests = tables[name]
        significant = tests[tests["p"] < 0.05]
        logger.info("%s: %d t-tests, %d with p<0.05", name, len(tests), len(significant))
        for row in significant.itertuples():
            logger.info(
                "t p<0.05: %s %s vs %s diff=%.3f t=%.3f p=%.4f",
                row.element, row.persona_a, row.persona_b, row.mean_diff, row.t, row.p,
            )
    logger.info("overall mean repeat sd=%.4f", tables["repeat_consistency.csv"].iloc[0]["mean_sd"])


def main() -> None:
    """Parse arguments and run the analysis."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
