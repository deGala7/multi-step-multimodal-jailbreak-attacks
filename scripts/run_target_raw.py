#!/usr/bin/env python3

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


API_KEY = os.environ.get("GEMINI_API_KEY")
DATASET_PATH = Path("experiment/data/harmful_dataset.csv")
RESPONSES_ROOT = Path("experiment/generated/responses/raw_baseline")
GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_RETRIES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send raw single-step dataset goals to Gemini one-by-one."
    )
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="Path to the harmful dataset CSV.")
    parser.add_argument(
        "--responses-root",
        default=str(RESPONSES_ROOT),
        help="Root folder where raw baseline responses are written.",
    )
    parser.add_argument("--model", default=GEMINI_MODEL, help="Gemini model name.")
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Generation temperature."
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum output tokens per request.",
    )
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries per request.")
    parser.add_argument(
        "--index",
        action="append",
        default=[],
        help="Specific index to process. Repeat to pass multiple indices.",
    )
    parser.add_argument(
        "--repair-broken",
        action="store_true",
        help="Process only missing/error raw outputs.",
    )
    parser.add_argument(
        "--reprocess-all",
        action="store_true",
        help="Ignore existing responses and regenerate every selected index.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def is_nonempty_text_file(path: Path) -> bool:
    return path.exists() and path.is_file() and bool(read_text(path).strip())


def load_dataset(dataset_path: Path) -> list[dict[str, str]]:
    with dataset_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    return [
        {"index": str(row.get("Index", "")).strip(), "goal": str(row.get("Goal", "")).strip()}
        for row in rows
        if str(row.get("Index", "")).strip()
    ]


def index_is_complete(index_dir: Path) -> bool:
    if (index_dir / "error.txt").exists():
        return False
    return is_nonempty_text_file(index_dir / "response.txt") and (index_dir / "response.json").exists()


def call_gemini(
    *,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "topP": 1,
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        body_text = err.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {err.code}: {body_text}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"Network error: {err}") from err


def extract_response_text(response_json: dict) -> str:
    candidates = response_json.get("candidates", [])
    if not candidates:
        raise ValueError(f"Gemini response missing candidates: {response_json}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
    if not text:
        raise ValueError(f"Gemini response did not contain text: {response_json}")
    return text


def selected(values: list[str], candidate: str) -> bool:
    return not values or candidate in values


def main() -> None:
    args = parse_args()
    api_key = API_KEY
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY environment variable.")

    rows = load_dataset(Path(args.dataset))
    responses_root = Path(args.responses_root)
    ensure_dir(responses_root)

    processed = 0
    failed = 0

    for row in rows:
        index_name = row["index"]
        goal = row["goal"]
        if not goal or not selected(args.index, index_name):
            continue

        output_dir = responses_root / index_name
        ensure_dir(output_dir)

        if not args.reprocess_all:
            complete = index_is_complete(output_dir)
            if args.repair_broken and complete:
                continue
            if not args.repair_broken and complete:
                print(f"[{index_name}] SKIP (already complete)")
                continue

        last_err = None
        for attempt in range(1, args.retries + 1):
            try:
                write_json(output_dir / "request.json", {"contents": [{"role": "user", "parts": [{"text": goal}]}]})
                response_json = call_gemini(
                    api_key=api_key,
                    model=args.model,
                    prompt=goal,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
                response_text = extract_response_text(response_json)
                write_text(output_dir / "request_prompt.txt", goal)
                write_text(output_dir / "response.txt", response_text)
                write_json(output_dir / "response.json", response_json)
                error_file = output_dir / "error.txt"
                if error_file.exists():
                    error_file.unlink()
                print(f"[{index_name}] OK")
                processed += 1
                last_err = None
                break
            except Exception as err:
                last_err = err
                if attempt < args.retries:
                    time.sleep(min(2**attempt, 10))

        if last_err is not None:
            write_text(output_dir / "error.txt", str(last_err))
            print(f"[{index_name}] FAILED: {last_err}")
            failed += 1

    print(f"Done. Processed: {processed}, Failed: {failed}")


if __name__ == "__main__":
    main()
