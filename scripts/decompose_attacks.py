#!/usr/bin/env python3

import argparse
import csv
import json
import os
import time
from pathlib import Path

from groq import Groq


API_KEY = os.environ.get("GROQ_API_KEY")
DATASET_PATH = Path("experiment/data/harmful_dataset.csv")
TEMPLATE_PATH = Path("experiment/templates/attack_decomposition_prompt.txt")
OUTPUT_DIR = Path("experiment/generated/decomposition/harmful")
MODEL = "llama-3.1-8b-instant"
MAX_COMPLETION_TOKENS = 4096
TEMPERATURE = 0.2
SLEEP_SECONDS = 0.0
RETRIES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose harmful goals into structured attack scenarios using Groq."
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Path to the harmful dataset CSV.",
    )
    parser.add_argument(
        "--template",
        default=str(TEMPLATE_PATH),
        help="Path to the attack decomposition prompt template.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Folder where decomposition outputs are written.",
    )
    parser.add_argument("--model", default=MODEL, help="Groq model name.")
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=MAX_COMPLETION_TOKENS,
        help="Maximum completion tokens for each request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=SLEEP_SECONDS,
        help="Delay between requests.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=RETRIES,
        help="Retries per failed request.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based row offset to start from.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to process.",
    )
    return parser.parse_args()


def call_groq(
    *,
    client: Groq,
    model: str,
    prompt: str,
    max_completion_tokens: int,
    temperature: float,
) -> tuple[dict, str]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=1,
            stream=False,
            stop=None,
        )
    except Exception as err:
        raise RuntimeError(str(err)) from err

    response_json = response.model_dump() if hasattr(response, "model_dump") else {}
    content = extract_content(response_json)
    return response_json, content


def extract_content(response_json: dict) -> str:
    choices = response_json.get("choices", [])
    if not choices:
        raise ValueError("Missing choices in Grok response JSON.")
    message = choices[0].get("message", {})
    content = message.get("content")
    if content is None:
        raise ValueError("Missing message.content in Grok response JSON.")
    return content


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    template_path = Path(args.template)
    output_dir = Path(args.output_dir)
    if not API_KEY:
        raise ValueError("Missing GROQ_API_KEY environment variable.")

    client = Groq(api_key=API_KEY)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    template_text = template_path.read_text(encoding="utf-8")
    if "__raw__prompt__" not in template_text:
        raise ValueError("Template must contain __raw__prompt__ placeholder.")

    output_dir.mkdir(parents=True, exist_ok=True)

    with dataset_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    required_columns = {"Index", "Goal"}
    if not rows:
        raise ValueError("Dataset CSV is empty.")
    if not required_columns.issubset(rows[0].keys()):
        raise ValueError(
            f"Dataset CSV must include columns: {sorted(required_columns)}"
        )

    selected_rows = rows[args.start :]
    if args.limit is not None:
        selected_rows = selected_rows[: args.limit]

    total = len(selected_rows)
    print(f"Processing {total} rows...")

    for i, row in enumerate(selected_rows, start=1):
        idx = str(row["Index"]).strip()
        goal = str(row["Goal"]).strip()
        prompt = template_text.replace("__raw__prompt__", goal)

        row_dir = output_dir / idx
        row_dir.mkdir(parents=True, exist_ok=True)
        write_text(row_dir / "request_prompt.txt", prompt)

        attempt = 0
        while True:
            try:
                response_json, content = call_groq(
                    client=client,
                    model=args.model,
                    prompt=prompt,
                    max_completion_tokens=args.max_completion_tokens,
                    temperature=args.temperature,
                )

                write_json(row_dir / "api_response.json", response_json)
                write_text(row_dir / "response_raw.txt", content)

                try:
                    parsed_content = json.loads(content)
                    write_json(row_dir / "response_parsed.json", parsed_content)
                except json.JSONDecodeError:
                    # Keep raw output only if model did not return valid JSON.
                    pass

                print(f"[{i}/{total}] Index {idx}: OK")
                break

            except (
                TimeoutError,
                ValueError,
                RuntimeError,
            ) as err:
                err_text = str(err)
                if "Incorrect API key provided" in err_text or "401" in err_text:
                    error_path = row_dir / "error.txt"
                    write_text(error_path, err_text)
                    raise RuntimeError(
                        "Authentication failed. Set GROQ_API_KEY to a valid Groq key."
                    ) from err

                if "403" in err_text and "credits" in err_text.lower():
                    error_path = row_dir / "error.txt"
                    write_text(error_path, err_text)
                    raise RuntimeError(
                        "Groq account has no credits or permission for this model. Add credits or switch model."
                    ) from err

                attempt += 1
                if attempt > args.retries:
                    error_path = row_dir / "error.txt"
                    write_text(error_path, err_text)
                    print(f"[{i}/{total}] Index {idx}: FAILED ({err_text})")
                    break
                wait_seconds = min(2**attempt, 10)
                print(
                    f"[{i}/{total}] Index {idx}: retry {attempt}/{args.retries} after error: {err_text}"
                )
                time.sleep(wait_seconds)

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    print("Done.")


if __name__ == "__main__":
    main()
