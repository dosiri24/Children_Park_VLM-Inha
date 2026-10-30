"""Apply per-element Bonferroni correction to the paired t-tests of the 53-park analysis and write CSV copies and a report."""

import argparse
import csv
import logging
from pathlib import Path

import pandas as pd
from scipy import stats


ANALYSIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ANALYSIS_DIR / "results"
SCORES_PATH = ANALYSIS_DIR.parent / "results" / "scores.csv"
PROJECT_DIR = ANALYSIS_DIR.parents[1]
ALPHA = 0.05
# Bonferroni family size per element: 6 child conditions vs baseline, C(6,2) child pairs.
FAMILY_SIZE = {"paired_vs_baseline.csv": 6, "paired_child_pairs.csv": 15}
BASELINE = "none"
# quant.py writes p with 4 decimals, so a recomputed p must match within half a unit of the last digit.
P_ROUNDING = 0.5e-4
MINUS = "−"


def configure_logging() -> logging.Logger:
    """Log to the console only, so reruns create no new files."""

    logger = logging.getLogger("correction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def park_means(scores_path: Path) -> pd.DataFrame:
    """Average repeats into one value per park, persona, and element, as quant.py does."""

    scores = pd.read_csv(scores_path)
    cells = scores.groupby(["element", "park_id", "persona_id"])["score"].mean().reset_index()
    return cells.pivot_table(index=["element", "park_id"], columns="persona_id", values="score")


def recompute_p(wide: pd.DataFrame, element: str, persona_a: str, persona_b: str) -> dict:
    """Run the paired t-test of persona_a vs persona_b for one element on park means."""

    parks = wide.loc[element]
    result = stats.ttest_rel(parks[persona_a], parks[persona_b])
    return {"n": len(parks), "mean_diff": (parks[persona_a] - parks[persona_b]).mean(), "t": result.statistic, "p": result.pvalue}


def correct(name: str, source: Path, wide: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Return the CSV rows as text plus alpha_bonf, sig_raw, sig_bonf, after checking p against raw scores."""

    table = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    family = FAMILY_SIZE[name]
    sizes = table.groupby("element").size()
    if not (sizes == family).all():
        raise ValueError(f"{name}: expected {family} comparisons per element, got {sizes.to_dict()}")

    p_exact, max_gap = [], 0.0
    for row in table.itertuples():
        fresh = recompute_p(wide, row.element, row.persona_a, row.persona_b)
        gap = abs(fresh["p"] - float(row.p))
        max_gap = max(max_gap, gap)
        if gap > P_ROUNDING or abs(fresh["mean_diff"] - float(row.mean_diff)) > P_ROUNDING or fresh["n"] != int(row.n_parks):
            raise ValueError(f"{name}: recomputed test differs from CSV for {row.element} {row.persona_a} vs {row.persona_b}")
        p_exact.append(fresh["p"])
    table["p_exact"] = p_exact
    alpha = ALPHA / family
    table["alpha_bonf"] = f"{alpha:.6f}"
    table["sig_raw"] = [p < ALPHA for p in p_exact]
    table["sig_bonf"] = [p < alpha for p in p_exact]
    # The text p in the CSV is rounded; flag any row where rounding would change a verdict.
    csv_raw = table["p"].astype(float) < ALPHA
    csv_bonf = table["p"].astype(float) < alpha
    flips = int((csv_raw != table["sig_raw"]).sum() + (csv_bonf != table["sig_bonf"]).sum())
    logger.info(
        "%s: %d rows, alpha_bonf=%.6f, max |p_recomputed - p_csv|=%.2e, verdicts differing between exact and rounded p=%d",
        name, len(table), alpha, max_gap, flips,
    )
    logger.info("%s: sig_raw=%d sig_bonf=%d", name, table["sig_raw"].sum(), table["sig_bonf"].sum())
    return table


def write_copy(table: pd.DataFrame, path: Path) -> None:
    """Write the original columns plus the three new ones; the exact p is kept out of the copy."""

    out = table.drop(columns=["p_exact"])
    out.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")


def signed(value: float) -> str:
    """Format a difference with two decimals and an explicit sign."""

    text = f"{value:.2f}"
    if text in ("0.00", "-0.00"):
        return "0.00"
    return f"+{text}" if value > 0 else text.replace("-", MINUS)


def age_of(persona: str) -> str:
    """Return the age group token of a child persona id."""

    return persona.split("_")[1]


def label(persona: str) -> str:
    """Return a short Korean label such as 6세 남 for a child persona id."""

    _, age, gender = persona.split("_")
    return f"{age}세 {'남' if gender == 'male' else '여'}"


def short(persona: str) -> str:
    """Return a table-header label such as 6남."""

    _, age, gender = persona.split("_")
    return f"{age}{'남' if gender == 'male' else '여'}"


def is_same_age(row: pd.Series) -> bool:
    """Tell whether a child pair shares the age group (hence differs only in gender)."""

    return age_of(row["persona_a"]) == age_of(row["persona_b"])


def count_line(frame: pd.DataFrame, column: str) -> int:
    """Count True flags in a column."""

    return int(frame[column].sum())


def hypothesis1_section(base: pd.DataFrame, elements: list[tuple[str, str]], children: list[str]) -> list[str]:
    """Build the per-element x condition table and per-element counts for hypothesis 1."""

    lines = [
        "## 2. 가설 1: 어린이 조건 대 페르소나 없음 (요소마다 6회, 기준 p < 0.05/6 = 0.00833)",
        "",
        f"- 유의한 비교: 보정 전 {count_line(base, 'sig_raw')}/48 → 보정 후 {count_line(base, 'sig_bonf')}/48",
        "- 표의 값은 평균 차이(어린이 − 페르소나 없음)다. 굵은 글씨는 보정 후에도 유의(p<0.00833)한 칸이고, "
        "† 표시는 보정 전에는 유의했으나(p<0.05) 보정 후 유의하지 않은 칸이다.",
        "",
        "| 요소 | " + " | ".join(short(c) for c in children) + " |",
        "|---|" + "---|" * len(children),
    ]
    for element, element_ko in elements:
        cells = []
        for child in children:
            row = base[(base["element"] == element) & (base["persona_a"] == child)].iloc[0]
            text = signed(float(row["mean_diff"]))
            if row["sig_bonf"]:
                text = f"**{text}**"
            elif row["sig_raw"]:
                text = f"{text}†"
            cells.append(text)
        lines.append(f"| {element_ko} | " + " | ".join(cells) + " |")
    lines += ["", "요소별 유의 횟수(/6)와 조건별 유의 횟수(/8)다.", "", "| 요소 | 보정 전 | 보정 후 |", "|---|---|---|"]
    for element, element_ko in elements:
        part = base[base["element"] == element]
        lines.append(f"| {element_ko} | {count_line(part, 'sig_raw')} | {count_line(part, 'sig_bonf')} |")
    lines += ["", "| 조건 | 보정 전 | 보정 후 |", "|---|---|---|"]
    for child in children:
        part = base[base["persona_a"] == child]
        lines.append(f"| {label(child)} | {count_line(part, 'sig_raw')} | {count_line(part, 'sig_bonf')} |")
    return lines + [""]


def hypothesis2_section(pairs: pd.DataFrame, elements: list[tuple[str, str]]) -> list[str]:
    """Build counts, same-age gender pair detail, and per-element counts for hypothesis 2."""

    same = pairs[pairs.apply(is_same_age, axis=1)]
    diff_age = pairs[~pairs.apply(is_same_age, axis=1)]
    lines = [
        "## 3. 가설 2: 어린이 조건끼리 (요소마다 15쌍, 기준 p < 0.05/15 = 0.00333)",
        "",
        "| 구분 | 보정 전 | 보정 후 |",
        "|---|---|---|",
        f"| 전체 | {count_line(pairs, 'sig_raw')}/120 | {count_line(pairs, 'sig_bonf')}/120 |",
        f"| 연령이 다른 쌍 | {count_line(diff_age, 'sig_raw')}/{len(diff_age)} | {count_line(diff_age, 'sig_bonf')}/{len(diff_age)} |",
        f"| 같은 연령 남녀 쌍 | {count_line(same, 'sig_raw')}/{len(same)} | {count_line(same, 'sig_bonf')}/{len(same)} |",
        "",
        "보정 후에도 유의한 같은 연령 남녀 쌍이다. 차이는 앞 조건 − 뒤 조건(남 − 여) 기준이다.",
        "",
        "| 연령 | 요소 | 차이 | p |",
        "|---|---|---|---|",
    ]
    for row in same[same["sig_bonf"]].itertuples():
        lines.append(f"| {age_of(row.persona_a)}세 | {row.element_ko} | {signed(float(row.mean_diff))} | {float(row.p):.4f} |")
    lines += ["", "요소별 유의 횟수(/15)다.", "", "| 요소 | 보정 전 | 보정 후 |", "|---|---|---|"]
    for element, element_ko in elements:
        part = pairs[pairs["element"] == element]
        lines.append(f"| {element_ko} | {count_line(part, 'sig_raw')} | {count_line(part, 'sig_bonf')} |")
    return lines + [""]


def lost_section(base: pd.DataFrame, pairs: pd.DataFrame) -> list[str]:
    """List comparisons significant before correction but not after."""

    lines = ["## 4. 보정 전에는 유의했으나 보정 후 유의하지 않은 비교", ""]
    for title, frame, alpha in (("가설 1", base, ALPHA / 6), ("가설 2", pairs, ALPHA / 15)):
        lost = frame[frame["sig_raw"] & ~frame["sig_bonf"]]
        lines += [
            f"### {title}: {len(lost)}회 (보정 후 기준 p < {alpha:.5f})",
            "",
            "| 요소 | 조건 | 차이 | p |",
            "|---|---|---|---|",
        ]
        for row in lost.itertuples():
            condition = f"{label(row.persona_a)} − 페르소나 없음" if row.persona_b == BASELINE else f"{label(row.persona_a)} − {label(row.persona_b)}"
            lines.append(f"| {row.element_ko} | {condition} | {signed(float(row.mean_diff))} | {float(row.p):.4f} |")
        lines.append("")
    return lines


def report(base: pd.DataFrame, pairs: pd.DataFrame, scores_path: Path) -> str:
    """Compose bonferroni_results.md."""

    elements = list(dict.fromkeys(zip(base["element"], base["element_ko"])))
    children = list(dict.fromkeys(base["persona_a"]))
    lines = [
        "# 53곳 전수 분석: 요소별 Bonferroni 보정 결과",
        "",
        f"- 원자료: `{scores_path.relative_to(PROJECT_DIR).as_posix()}`에서 공원×조건×요소 5회 평균을 다시 계산해 "
        "`paired_vs_baseline.csv`·`paired_child_pairs.csv`의 p를 대조했다(전 행 소수 넷째 자리까지 일치).",
        "- 방법: 요소마다 비교 묶음을 따로 잡고 유의 기준을 0.05/묶음 크기로 나눈다. 가설 1은 요소마다 6회(어린이 6조건 대 페르소나 없음)라 "
        "p<0.05/6, 가설 2는 요소마다 15쌍(어린이 6조건끼리)이라 p<0.05/15다. 보정은 두 가설에서 각각 따로 한다.",
        "- 판정에는 재계산한 정확한 p를 썼다. CSV의 p는 소수 넷째 자리 반올림값이다.",
        "- 파일: `paired_vs_baseline.csv`, `paired_child_pairs.csv`는 원본 열 뒤에 `alpha_bonf`, `sig_raw`(p<0.05), `sig_bonf`를 더한 사본이다.",
        "",
        "## 1. 핵심 숫자 (보정 전 → 보정 후)",
        "",
    ]
    same = pairs[pairs.apply(is_same_age, axis=1)]
    diff_age = pairs[~pairs.apply(is_same_age, axis=1)]
    lines += [
        "| 항목 | 보정 전 | 보정 후 |",
        "|---|---|---|",
        f"| 가설 1 유의 횟수 | {count_line(base, 'sig_raw')}/48 | {count_line(base, 'sig_bonf')}/48 |",
        f"| 가설 2 유의 횟수 | {count_line(pairs, 'sig_raw')}/120 | {count_line(pairs, 'sig_bonf')}/120 |",
        f"| 가설 2 연령이 다른 쌍 | {count_line(diff_age, 'sig_raw')}/96 | {count_line(diff_age, 'sig_bonf')}/96 |",
        f"| 가설 2 같은 연령 남녀 쌍 | {count_line(same, 'sig_raw')}/24 | {count_line(same, 'sig_bonf')}/24 |",
        "",
    ]
    lines += hypothesis1_section(base, elements, children)
    lines += hypothesis2_section(pairs, elements)
    lines += lost_section(base, pairs)
    return "\n".join(lines).rstrip("\n") + "\n"


def run() -> None:
    """Correct both paired-test tables and write the copies and the report."""

    logger = configure_logging()
    results_dir = RESULTS_DIR
    out_dir = RESULTS_DIR / "bonferroni"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = SCORES_PATH

    wide = park_means(scores_path)
    logger.info("loaded park means from %s: %d park x element rows", scores_path, len(wide))
    tables = {name: correct(name, results_dir / name, wide, logger) for name in FAMILY_SIZE}
    for name, table in tables.items():
        write_copy(table, out_dir / name)
        logger.info("wrote %s", out_dir / name)

    text = report(tables["paired_vs_baseline.csv"], tables["paired_child_pairs.csv"], scores_path)
    (out_dir / "bonferroni_results.md").write_text(text, encoding="utf-8", newline="\n")
    logger.info("wrote %s", out_dir / "bonferroni_results.md")


def main() -> None:
    """Parse arguments and run the correction."""

    argparse.ArgumentParser(description=__doc__).parse_args()
    run()


if __name__ == "__main__":
    main()
