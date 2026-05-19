#!/usr/bin/env python3

import argparse
from pathlib import Path

from common.judge_statistics_common import compute_statistics, write_report


JUDGE_ROOT = Path("experiment/results/judge/raw_baseline")
DATASET_PATH = Path("experiment/data/harmful_dataset.csv")
OUTPUT_PATH = Path("experiment/results/statistics/raw_baseline.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute detailed statistics from raw baseline judge outputs."
    )
    parser.add_argument(
        "--judge-root",
        default=str(JUDGE_ROOT),
        help="Path to raw baseline judge output directory.",
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
        layout="flat",
    )
    output_path = Path(args.output)
    write_report(output_path, report)

    summary = report["summary"]
    print(f"Wrote statistics to {output_path}")
    print(
        "Summary: "
        f"successful={summary['successful_judgements']}, "
        f"failed={summary['failed_units']}, "
        f"invalid={summary['invalid_outputs']}, "
        f"missing={summary['missing_outputs']}"
    )


if __name__ == "__main__":
    main()
