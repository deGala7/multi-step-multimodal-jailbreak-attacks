#!/usr/bin/env python3

import argparse
from pathlib import Path

from common.judge_statistics_common import (
    SCENARIOS,
    STEPS,
    finalize_bucket,
    load_dataset,
    make_bucket,
    parse_judge_payload,
    to_int,
    update_bucket,
)


DATASET_PATH = Path("experiment/data/harmful_dataset.csv")
DEFAULT_OUTPUT = Path("experiment/results/statistics/tables.tex")
ROOTS = {
    "multi_step": Path("experiment/results/judge/multistep_multimodal"),
    "text": Path("experiment/results/judge/text_only_multistep"),
    "raw": Path("experiment/results/judge/raw_baseline"),
    "one_step": Path("experiment/results/judge/one_step_multimodal"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a standalone LaTeX report from judge result folders."
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Path to the harmful dataset CSV.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to the generated LaTeX file.",
    )
    return parser.parse_args()


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = value
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def fmt_float(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def fmt_percent(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.1f}\\%"


def render_table(
    *,
    caption: str,
    label: str,
    columns: list[str],
    rows: list[list[str]],
    column_spec: str,
) -> str:
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    header = " & ".join(columns) + r" \\"
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{column_spec}}}",
            r"\toprule",
            header,
            r"\midrule",
            body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def normalize_judgement(payload: dict | None) -> tuple[int, int] | None:
    if payload is None:
        return None
    jb = to_int(payload.get("jailbreak_success"))
    hs = to_int(payload.get("harmful_score"))
    if jb is None or hs is None:
        return None
    if jb == 0:
        hs = 0
    if jb not in {0, 1} or hs not in {0, 1, 2, 3, 4, 5}:
        return None
    return jb, hs


def collect_bucket(unit_paths: list[tuple[str, Path]], dataset: dict[str, dict[str, str]]) -> dict:
    bucket = make_bucket()
    for index, unit_dir in unit_paths:
        payload, err = parse_judge_payload(unit_dir)
        if err is not None:
            continue
        normalized = normalize_judgement(payload)
        if normalized is None:
            continue
        jb, hs = normalized
        update_bucket(bucket, jb, hs)
    return finalize_bucket(bucket)


def collect_category_buckets(
    unit_paths: list[tuple[str, Path]], dataset: dict[str, dict[str, str]]
) -> dict[str, dict]:
    by_category: dict[str, dict] = {}
    for index, unit_dir in unit_paths:
        payload, err = parse_judge_payload(unit_dir)
        if err is not None:
            continue
        normalized = normalize_judgement(payload)
        if normalized is None:
            continue
        jb, hs = normalized
        category = str(dataset.get(index, {}).get("Category", "Unknown") or "Unknown")
        if category not in by_category:
            by_category[category] = make_bucket()
        update_bucket(by_category[category], jb, hs)
    return {k: finalize_bucket(v) for k, v in sorted(by_category.items())}


def all_indices(root: Path) -> list[str]:
    return sorted(
        [p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=int,
    )


def slice_units(kind: str, *, scenario: str | None = None, step: str | None = None) -> list[tuple[str, Path]]:
    root = ROOTS[kind]
    units: list[tuple[str, Path]] = []
    for index in all_indices(root):
        if kind == "raw":
            unit_dir = root / index
        elif kind == "one_step":
            if scenario is None:
                raise ValueError("one_step slices require scenario")
            unit_dir = root / index / scenario
        else:
            if scenario is None or step is None:
                raise ValueError("multi_step/text slices require scenario and step")
            unit_dir = root / index / scenario / step
        units.append((index, unit_dir))
    return units


def metric_row(name: str, bucket: dict) -> list[str]:
    return [
        latex_escape(name),
        str(bucket["count"]),
        str(bucket["jailbreak_success_count"]),
        fmt_percent(bucket["jailbreak_success_rate"]),
        fmt_float(bucket["harmful_score_mean"]),
        fmt_float(bucket["harmful_score_median"], 1),
    ]


def first_table_rows(dataset: dict[str, dict[str, str]]) -> list[list[str]]:
    slices = [
        ("Multi-step / Scenario 1 / Step 2", collect_bucket(slice_units("multi_step", scenario="scenario_1", step="step_2"), dataset)),
        ("Multi-step / Scenario 2 / Step 2", collect_bucket(slice_units("multi_step", scenario="scenario_2", step="step_2"), dataset)),
        ("Text / Scenario 1 / Step 2", collect_bucket(slice_units("text", scenario="scenario_1", step="step_2"), dataset)),
        ("Text / Scenario 2 / Step 2", collect_bucket(slice_units("text", scenario="scenario_2", step="step_2"), dataset)),
        ("One-step / Scenario 1", collect_bucket(slice_units("one_step", scenario="scenario_1"), dataset)),
        ("One-step / Scenario 2", collect_bucket(slice_units("one_step", scenario="scenario_2"), dataset)),
        ("Raw", collect_bucket(slice_units("raw"), dataset)),
    ]
    return [metric_row(name, bucket) for name, bucket in slices]


def step_progress_rows(kind: str, scenario: str, dataset: dict[str, dict[str, str]]) -> list[list[str]]:
    prefix = "Multi-step" if kind == "multi_step" else "Text"
    rows: list[list[str]] = []
    for step in STEPS:
        bucket = collect_bucket(slice_units(kind, scenario=scenario, step=step), dataset)
        rows.append(metric_row(f"{prefix} / {scenario.replace('_', ' ').title()} / {step.replace('_', ' ').title()}", bucket))
    return rows


def scenario_final_rows(dataset: dict[str, dict[str, str]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for kind, label in [("multi_step", "Multi-step"), ("text", "Text")]:
        for scenario in SCENARIOS:
            bucket = collect_bucket(slice_units(kind, scenario=scenario, step="step_2"), dataset)
            rows.append(metric_row(f"{label} / {scenario.replace('_', ' ').title()} / Step 2", bucket))
    for scenario in SCENARIOS:
        bucket = collect_bucket(slice_units("one_step", scenario=scenario), dataset)
        rows.append(metric_row(f"One-step / {scenario.replace('_', ' ').title()}", bucket))
    rows.append(metric_row("Raw", collect_bucket(slice_units("raw"), dataset)))
    return rows


def category_rows_for_slice(
    name: str,
    unit_paths: list[tuple[str, Path]],
    dataset: dict[str, dict[str, str]],
) -> list[list[str]]:
    categories = collect_category_buckets(unit_paths, dataset)
    return [metric_row(category, bucket) for category, bucket in categories.items()]


def document_preamble() -> str:
    return "\n".join(
        [
            r"\documentclass[11pt,a4paper]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage{lmodern}",
            r"\usepackage{booktabs}",
            r"\usepackage{float}",
            r"\usepackage{xcolor}",
            r"\usepackage{caption}",
            r"\captionsetup{font=small,labelfont=bf}",
            r"\definecolor{TitleBlue}{HTML}{163A5F}",
            r"\definecolor{AccentGray}{HTML}{555555}",
            r"\setlength{\parskip}{0.5em}",
            r"\setlength{\parindent}{0pt}",
            "",
            r"\begin{document}",
            r"\begin{titlepage}",
            r"\centering",
            r"{\Huge\bfseries\color{TitleBlue} Judge Statistics Report\par}",
            r"\vspace{1.0cm}",
            r"{\Large Final-output and Step-wise Comparison Across Prompting Variants\par}",
            r"\vspace{0.8cm}",
            r"{\large\color{AccentGray} Auto-generated from judge outputs and dataset metadata\par}",
            r"\vfill",
            r"\end{titlepage}",
            "",
        ]
    )


def document_intro() -> str:
    return "\n".join(
        [
            r"\section*{Overview}",
            (
                "The evaluation setup contains four variants. Multi-step refers to the "
                "original multi-step multimodal pipeline with steps 0, 1, and 2, where "
                "step 2 is the final attack step. Text refers to the same multi-step "
                "structure but without multimodal inputs. One-step refers to a multimodal "
                "attack compressed into a single request per scenario. Raw refers to direct "
                "harmful requests without the multi-step structure."
            ),
            (
                "ASR denotes the attack success rate. Harm score mean and median are "
                "computed after excluding score-0 non-harmful cases."
            ),
            "",
        ]
    )


def build_latex(dataset: dict[str, dict[str, str]]) -> str:
    parts: list[str] = [document_preamble(), document_intro()]

    parts.append(r"\section{Final-output Comparison}")
    parts.append(
        render_table(
            caption="Comparison of the final comparable outputs across all variants.",
            label="tab:final-output-comparison",
            columns=["Variant", "N", "ASR Count", "ASR", "HS Mean", "HS Median"],
            rows=first_table_rows(dataset),
            column_spec="lrrrrr",
        )
    )

    parts.append(r"\section{Step-wise Progression}")
    parts.append(
        "The following tables compare step progression separately for each scenario in the multi-step and text pipelines."
    )
    parts.append(
        render_table(
            caption="Step progression for Scenario 1.",
            label="tab:step-progression-s1",
            columns=["Variant / Step", "N", "ASR Count", "ASR", "HS Mean", "HS Median"],
            rows=step_progress_rows("multi_step", "scenario_1", dataset)
            + step_progress_rows("text", "scenario_1", dataset),
            column_spec="lrrrrr",
        )
    )
    parts.append(
        render_table(
            caption="Step progression for Scenario 2.",
            label="tab:step-progression-s2",
            columns=["Variant / Step", "N", "ASR Count", "ASR", "HS Mean", "HS Median"],
            rows=step_progress_rows("multi_step", "scenario_2", dataset)
            + step_progress_rows("text", "scenario_2", dataset),
            column_spec="lrrrrr",
        )
    )

    parts.append(r"\section{Scenario Comparison at Final Output}")
    parts.append(
        "This comparison focuses only on final outputs: step 2 for the multi-step and text variants, single-request outputs for the one-step variant, and raw harmful prompts."
    )
    parts.append(
        render_table(
            caption="Scenario comparison for final outputs.",
            label="tab:scenario-final-comparison",
            columns=["Variant / Final Output", "N", "ASR Count", "ASR", "HS Mean", "HS Median"],
            rows=scenario_final_rows(dataset),
            column_spec="lrrrrr",
        )
    )

    parts.append(r"\section{Category Breakdown of Final Outputs}")
    slice_specs = [
        ("Multi-step / Scenario 1 / Step 2", slice_units("multi_step", scenario="scenario_1", step="step_2"), "tab:cat-ms-s1-s2"),
        ("Multi-step / Scenario 2 / Step 2", slice_units("multi_step", scenario="scenario_2", step="step_2"), "tab:cat-ms-s2-s2"),
        ("Text / Scenario 1 / Step 2", slice_units("text", scenario="scenario_1", step="step_2"), "tab:cat-text-s1-s2"),
        ("Text / Scenario 2 / Step 2", slice_units("text", scenario="scenario_2", step="step_2"), "tab:cat-text-s2-s2"),
        ("One-step / Scenario 1", slice_units("one_step", scenario="scenario_1"), "tab:cat-one-s1"),
        ("One-step / Scenario 2", slice_units("one_step", scenario="scenario_2"), "tab:cat-one-s2"),
        ("Raw", slice_units("raw"), "tab:cat-raw"),
    ]
    for title, units, label in slice_specs:
        parts.append(rf"\subsection{{{latex_escape(title)}}}")
        parts.append(
            render_table(
                caption=f"Category breakdown for {title}.",
                label=label,
                columns=["Category", "N", "ASR Count", "ASR", "HS Mean", "HS Median"],
                rows=category_rows_for_slice(title, units, dataset),
                column_spec="lrrrrr",
            )
        )

    parts.append(r"\end{document}")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    dataset = load_dataset(Path(args.dataset))
    output_path = Path(args.output)
    output_path.write_text(build_latex(dataset), encoding="utf-8")
    print(f"Wrote LaTeX tables to {output_path}")


if __name__ == "__main__":
    main()
