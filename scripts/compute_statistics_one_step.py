#!/usr/bin/env python3

import argparse
from pathlib import Path

from common.judge_statistics_common import compute_statistics, write_report


JUDGE_ROOT = Path("experiment/results/judge/one_step_multimodal")
DATASET_PATH = Path("experiment/data/harmful_dataset.csv")
OUTPUT_PATH = Path("experiment/results/statistics/one_step_multimodal.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute detailed statistics from one-step judge outputs."
    )
    parser.add_argument(
        "--judge-root",
        default=str(JUDGE_ROOT),
        help="Path to one-step judge output directory.",
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


def main() -> None:
    args = parse_args()
    report = compute_statistics(
        judge_root=Path(args.judge_root),
        dataset_path=Path(args.dataset),
        layout="scenario",
    )
    output_path = Path(args.output)
    write_report(output_path, report)

    summary = report["summary"]
    print(f"Wrote statistics to {output_path}")
    print(
        "Summary: "
        f"successful={summary['successful_judgements']}, "
        f"failed={summary['failed_scenarios']}, "
        f"invalid={summary['invalid_outputs']}, "
        f"missing={summary['missing_outputs']}"
    )


if __name__ == "__main__":
    main()
