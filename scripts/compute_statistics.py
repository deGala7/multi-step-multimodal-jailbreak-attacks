#!/usr/bin/env python3

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


JUDGE_ROOT = Path("experiment/results/judge/multistep_multimodal")
DATASET_PATH = Path("experiment/data/harmful_dataset.csv")
OUTPUT_PATH = Path("experiment/results/statistics/multistep_multimodal.json")
SCENARIOS = ["scenario_1", "scenario_2"]
STEPS = ["step_0", "step_1", "step_2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute detailed statistics from judge folder outputs."
    )
    parser.add_argument(
        "--judge-root",
        default=str(JUDGE_ROOT),
        help="Path to judge root directory.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Path to dataset CSV (optional metadata enrichment).",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Path for output JSON report.",
    )
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> dict[str, dict[str, str]]:
    if not dataset_path.exists():
        return {}

    with dataset_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    data: dict[str, dict[str, str]] = {}
    for row in rows:
        index = str(row.get("Index", "")).strip()
        if index:
            data[index] = row
    return data


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def parse_first_json_object(text: str) -> dict | None:
    candidate = extract_json_text(text)

    # Fast path for strict JSON content.
    try:
        parsed = json.loads(candidate, strict=False)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Recovery path: find first decodable JSON object inside mixed text.
    decoder = json.JSONDecoder(strict=False)
    for i, ch in enumerate(candidate):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(candidate[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    # Final fallback: salvage the judge fields from malformed text when the
    # model emitted the expected keys but failed to close the analysis string.
    jail_match = re.search(r'"jailbreak_success"\s*:\s*([01])', candidate)
    harm_match = re.search(r'"harmful_score"\s*:\s*([0-5])', candidate)
    if jail_match and harm_match:
        return {
            "jailbreak_success": int(jail_match.group(1)),
            "harmful_score": int(harm_match.group(1)),
        }

    return None


def parse_judge_payload(step_dir: Path) -> tuple[dict | None, str | None]:
    response_txt = step_dir / "response.txt"
    if response_txt.exists():
        raw = read_text(response_txt)
        parsed = parse_first_json_object(raw)
        if parsed is not None:
            return parsed, None

    response_json = step_dir / "response.json"
    if response_json.exists():
        try:
            outer = json.loads(read_text(response_json))
        except json.JSONDecodeError as err:
            return None, f"invalid response.json: {err}"

        content = outer.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content:
            parsed = parse_first_json_object(content)
            if parsed is not None:
                return parsed, None
            return None, "response content is not valid JSON"

    return None, "missing or unparsable judge output"


def to_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if s in {"0", "1", "2", "3", "4", "5"}:
            return int(s)
    if isinstance(value, list) and value:
        return to_int(value[0])
    return None


def normalize_judgement(
    jailbreak_success: int | None, harmful_score: int | None
) -> tuple[int | None, int | None]:
    if jailbreak_success == 0:
        return 0, 0
    return jailbreak_success, harmful_score


def update_bucket(bucket: dict, jb: int, hs: int) -> None:
    bucket["count"] += 1
    bucket["jailbreak_success_count"] += jb
    bucket["harmful_scores"].append(hs)
    bucket["harmful_score_distribution"][str(hs)] += 1


def finalize_bucket(bucket: dict) -> dict:
    count = bucket["count"]
    if count == 0:
        return {
            "count": 0,
            "jailbreak_success_count": 0,
            "jailbreak_success_rate": 0.0,
            "harmful_score_mean": None,
            "harmful_score_median": None,
            "harmful_score_distribution": dict(bucket["harmful_score_distribution"]),
        }

    harmful_scores = bucket["harmful_scores"]
    harmful_scores_for_stats = [score for score in harmful_scores if score > 0]
    return {
        "count": count,
        "jailbreak_success_count": bucket["jailbreak_success_count"],
        "jailbreak_success_rate": bucket["jailbreak_success_count"] / count,
        "harmful_score_mean": (
            statistics.fmean(harmful_scores_for_stats)
            if harmful_scores_for_stats
            else None
        ),
        "harmful_score_median": (
            statistics.median(harmful_scores_for_stats)
            if harmful_scores_for_stats
            else None
        ),
        "harmful_score_distribution": dict(bucket["harmful_score_distribution"]),
    }


def make_bucket() -> dict:
    return {
        "count": 0,
        "jailbreak_success_count": 0,
        "harmful_scores": [],
        "harmful_score_distribution": Counter(),
    }


def main() -> None:
    args = parse_args()
    judge_root = Path(args.judge_root)
    dataset = load_dataset(Path(args.dataset))
    output_path = Path(args.output)

    if not judge_root.exists():
        raise FileNotFoundError(f"Judge root not found: {judge_root}")

    total_expected = len(dataset) * len(SCENARIOS) * len(STEPS) if dataset else None

    summary = {
        "total_expected_steps": total_expected,
        "total_found_step_dirs": 0,
        "successful_judgements": 0,
        "failed_steps": 0,
        "missing_outputs": 0,
        "invalid_outputs": 0,
    }

    error_reasons = Counter()
    overall_bucket = make_bucket()
    by_scenario: dict[str, dict] = defaultdict(make_bucket)
    by_step: dict[str, dict] = defaultdict(make_bucket)
    by_scenario_step: dict[str, dict] = defaultdict(make_bucket)
    by_category: dict[str, dict] = defaultdict(make_bucket)
    by_source: dict[str, dict] = defaultdict(make_bucket)

    for index_dir in sorted(judge_root.iterdir(), key=lambda p: p.name):
        if not index_dir.is_dir() or not index_dir.name.isdigit():
            continue
        index = index_dir.name

        for scenario in SCENARIOS:
            for step in STEPS:
                step_dir = index_dir / scenario / step
                if not step_dir.exists():
                    continue

                summary["total_found_step_dirs"] += 1

                error_file = step_dir / "error.txt"
                if error_file.exists():
                    summary["failed_steps"] += 1
                    reason = read_text(error_file).strip() or "unknown"
                    error_reasons[reason] += 1
                    continue

                payload, parse_error = parse_judge_payload(step_dir)
                if parse_error is not None:
                    if (step_dir / "response.txt").exists() or (
                        step_dir / "response.json"
                    ).exists():
                        summary["invalid_outputs"] += 1
                    else:
                        summary["missing_outputs"] += 1
                    error_reasons[parse_error] += 1
                    continue

                jb = to_int(payload.get("jailbreak_success"))
                hs = to_int(payload.get("harmful_score"))
                jb, hs = normalize_judgement(jb, hs)
                if jb is None or hs is None:
                    summary["invalid_outputs"] += 1
                    error_reasons["missing jailbreak_success or harmful_score"] += 1
                    continue

                summary["successful_judgements"] += 1
                update_bucket(overall_bucket, jb, hs)
                update_bucket(by_scenario[scenario], jb, hs)
                update_bucket(by_step[step], jb, hs)
                update_bucket(by_scenario_step[f"{scenario}/{step}"], jb, hs)

                if index in dataset:
                    category = str(
                        dataset[index].get("Category", "Unknown") or "Unknown"
                    )
                    source = str(dataset[index].get("Source", "Unknown") or "Unknown")
                    update_bucket(by_category[category], jb, hs)
                    update_bucket(by_source[source], jb, hs)

    report = {
        "summary": summary,
        "overall": finalize_bucket(overall_bucket),
        "by_scenario": {k: finalize_bucket(v) for k, v in sorted(by_scenario.items())},
        "by_step": {k: finalize_bucket(v) for k, v in sorted(by_step.items())},
        "by_scenario_step": {
            k: finalize_bucket(v) for k, v in sorted(by_scenario_step.items())
        },
        "by_category": {k: finalize_bucket(v) for k, v in sorted(by_category.items())},
        "by_source": {k: finalize_bucket(v) for k, v in sorted(by_source.items())},
        "top_error_reasons": [
            {"reason": reason, "count": count}
            for reason, count in error_reasons.most_common(20)
        ],
    }

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote statistics to {output_path}")
    print(
        "Summary: "
        f"successful={summary['successful_judgements']}, "
        f"failed={summary['failed_steps']}, "
        f"invalid={summary['invalid_outputs']}, "
        f"missing={summary['missing_outputs']}"
    )


if __name__ == "__main__":
    main()
