#!/usr/bin/env python3

import argparse
import csv
import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types


API_KEY = os.environ.get("GEMINI_API_KEY")
DATASET_PATH = Path("experiment/data/harmful_dataset.csv")
RESPONSES_ROOT = Path("experiment/generated/responses/raw_baseline")
MODEL = "gemini-3.1-pro-preview"
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Gemini Batch API for raw dataset prompts (single-step). "
            "Each Index/Goal is sent as one request and saved under raw baseline responses by index."
        )
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Path to the harmful dataset CSV.",
    )
    parser.add_argument(
        "--responses-root",
        default=str(RESPONSES_ROOT),
        help="Output folder for raw single-step responses.",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help="Gemini model for batch requests.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="Max output tokens per request.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect completed jobs, do not submit a new batch.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Include already completed outputs in the new batch.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def is_nonempty_text_file(path: Path) -> bool:
    return (
        path.exists()
        and path.is_file()
        and bool(path.read_text(encoding="utf-8").strip())
    )


def sanitize_json_value(value: object) -> object:
    if isinstance(value, bytes):
        return {
            "__type__": "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json_value(item) for item in value]
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(sanitize_json_value(value), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(dataset_path: Path) -> list[dict[str, str]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError("Dataset CSV is empty.")

    required = {"Index", "Goal"}
    if not required.issubset(rows[0].keys()):
        raise ValueError(f"Dataset CSV must include columns: {sorted(required)}")

    cleaned: list[dict[str, str]] = []
    for row in rows:
        cleaned.append(
            {
                "index": str(row.get("Index", "")).strip(),
                "goal": str(row.get("Goal", "")).strip(),
            }
        )
    return [item for item in cleaned if item["index"]]


def target_needs_processing(responses_root: Path, index_name: str) -> bool:
    out_dir = responses_root / index_name
    if not out_dir.exists():
        return True
    if (out_dir / "error.txt").exists():
        return True
    if not is_nonempty_text_file(out_dir / "response.txt"):
        return True
    return not (out_dir / "response.json").exists()


def build_inlined_requests(
    *,
    dataset_rows: list[dict[str, str]],
    responses_root: Path,
    model: str,
    temperature: float,
    max_output_tokens: int,
    reprocess: bool,
) -> tuple[list[types.InlinedRequest], list[dict]]:
    requests: list[types.InlinedRequest] = []
    manifest: list[dict] = []

    for row in dataset_rows:
        index_name = row["index"]
        goal = row["goal"]

        if not goal:
            continue

        if (not reprocess) and (
            not target_needs_processing(responses_root, index_name)
        ):
            continue

        metadata = {
            "index": index_name,
            "kind": "raw",
        }
        config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "top_p": 1,
        }
        contents = [{"role": "user", "parts": [{"text": goal}]}]

        requests.append(
            types.InlinedRequest(
                model=model,
                contents=contents,
                metadata=metadata,
                config=config,
            )
        )
        manifest.append({**metadata, "goal": goal})

    return requests, manifest


def jobs_dir(responses_root: Path) -> Path:
    return responses_root / "_batch_jobs"


def job_state_path(responses_root: Path) -> Path:
    return jobs_dir(responses_root) / "raw.json"


def get_job_state(responses_root: Path) -> dict | None:
    path = job_state_path(responses_root)
    if not path.exists():
        return None
    return load_json(path)


def set_job_state(responses_root: Path, state: dict) -> None:
    ensure_dir(jobs_dir(responses_root))
    write_json(job_state_path(responses_root), state)


def clear_error_if_present(output_dir: Path) -> None:
    error_path = output_dir / "error.txt"
    if error_path.exists():
        error_path.unlink()


def extract_response_text(response_obj: object) -> str:
    if response_obj is None:
        return ""

    if hasattr(response_obj, "candidates"):
        candidates = getattr(response_obj, "candidates") or []
        if candidates:
            first = candidates[0]
            content = getattr(first, "content", None)
            if content is not None:
                parts = getattr(content, "parts", []) or []
                chunks = []
                for part in parts:
                    text = getattr(part, "text", "")
                    if text:
                        chunks.append(text)
                return "".join(chunks).strip()

    if hasattr(response_obj, "text"):
        return str(getattr(response_obj, "text") or "").strip()

    return ""


def collect_job_results(
    *,
    client: genai.Client,
    responses_root: Path,
    job_state: dict,
) -> tuple[bool, str]:
    job_name = job_state.get("job_name", "")
    if not job_name:
        return True, "Invalid job state (missing job_name)."

    job = client.batches.get(name=job_name)
    state_name = getattr(job.state, "name", str(job.state))

    if state_name not in TERMINAL_STATES:
        return False, f"raw job still running: {state_name} ({job_name})"

    requests_manifest = job_state.get("requests", [])
    responses = []
    if job.dest and getattr(job.dest, "inlined_responses", None):
        responses = list(job.dest.inlined_responses)

    if state_name == "JOB_STATE_SUCCEEDED" and not responses:
        return True, f"raw job succeeded but returned no inlined responses ({job_name})"

    for i, item in enumerate(responses):
        metadata = getattr(item, "metadata", None) or {}
        if not metadata and i < len(requests_manifest):
            metadata = requests_manifest[i]

        index_name = str(metadata.get("index", ""))
        if not index_name:
            continue

        out_dir = responses_root / index_name
        ensure_dir(out_dir)

        goal = str(metadata.get("goal", ""))
        if goal:
            write_text(out_dir / "request_prompt.txt", goal)

        item_error = getattr(item, "error", None)
        if item_error:
            write_text(out_dir / "error.txt", str(item_error))
            continue

        response_obj = getattr(item, "response", None)
        text = extract_response_text(response_obj)
        if not text:
            write_text(
                out_dir / "error.txt", f"No text in batch response for {index_name}"
            )
            continue

        clear_error_if_present(out_dir)
        write_text(out_dir / "response.txt", text)
        if response_obj is not None and hasattr(response_obj, "model_dump"):
            write_json(out_dir / "response.json", response_obj.model_dump())

    job_state["state"] = state_name
    job_state["collected_at"] = now_iso()
    set_job_state(responses_root, job_state)
    return True, f"Collected raw job: {job_name} ({state_name})"


def any_non_terminal_job(responses_root: Path) -> bool:
    state = get_job_state(responses_root)
    if not state:
        return False
    current = str(state.get("state", ""))
    return current not in TERMINAL_STATES


def submit_batch(
    *,
    client: genai.Client,
    dataset_rows: list[dict[str, str]],
    responses_root: Path,
    model: str,
    temperature: float,
    max_output_tokens: int,
    reprocess: bool,
) -> None:
    if any_non_terminal_job(responses_root):
        print("Active non-terminal raw job already exists. Run again later.")
        return

    requests, manifest = build_inlined_requests(
        dataset_rows=dataset_rows,
        responses_root=responses_root,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        reprocess=reprocess,
    )

    if not requests:
        print("No requests to submit for raw prompts.")
        return

    print(f"Submitting raw batch: {len(requests)} requests")
    batch_job = client.batches.create(model=model, src=requests)
    job_name = str(batch_job.name)
    state_name = getattr(batch_job.state, "name", str(batch_job.state))

    set_job_state(
        responses_root,
        {
            "job_name": job_name,
            "state": state_name,
            "submitted_at": now_iso(),
            "request_count": len(requests),
            "requests": manifest,
        },
    )

    print(f"Submitted raw job: {job_name} ({state_name})")


def main() -> None:
    args = parse_args()
    api_key = API_KEY
    if not api_key:
        raise ValueError("Missing Gemini API key. Set GEMINI_API_KEY.")

    dataset_rows = load_dataset(Path(args.dataset))
    responses_root = Path(args.responses_root)
    ensure_dir(responses_root)
    ensure_dir(jobs_dir(responses_root))

    client = genai.Client(api_key=api_key)

    state = get_job_state(responses_root)
    if state:
        finished, message = collect_job_results(
            client=client,
            responses_root=responses_root,
            job_state=state,
        )
        print(message)
        if not finished:
            return

    if not args.collect_only:
        submit_batch(
            client=client,
            dataset_rows=dataset_rows,
            responses_root=responses_root,
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            reprocess=args.reprocess,
        )
    else:
        print("Collection pass done (collect-only mode).")


if __name__ == "__main__":
    main()
