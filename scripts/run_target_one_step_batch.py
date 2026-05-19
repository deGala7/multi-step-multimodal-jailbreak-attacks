#!/usr/bin/env python3

import argparse
import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types


API_KEY = os.environ.get("GEMINI_API_KEY")
PROMPTS_ROOT = Path("experiment/generated/prompts/multistep_multimodal")
RESPONSES_ROOT = Path("experiment/generated/responses/one_step_multimodal")
MODEL = "gemini-3.1-pro-preview"
STEPS = ["step_0", "step_1", "step_2"]
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
            "Run Gemini Batch API for one-step combined scenario prompts. "
            "For each index/scenario, all three steps are merged into one request "
            "and written under one-step responses by index and scenario."
        )
    )
    parser.add_argument(
        "--prompts-root",
        default=str(PROMPTS_ROOT),
        help="Root folder containing multimodal prompt assets by index, scenario, and step.",
    )
    parser.add_argument(
        "--responses-root",
        default=str(RESPONSES_ROOT),
        help="Root folder for one-step combined responses.",
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
        default=4096,
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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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


def list_index_dirs(prompts_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(prompts_root.iterdir(), key=lambda p: p.name)
        if path.is_dir()
    ]


def file_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".ogg":
        return "audio/ogg"
    raise ValueError(f"Unsupported media file type: {path}")


def inline_data_part(path: Path) -> dict:
    return {
        "inline_data": {
            "mime_type": file_mime_type(path),
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
    }


def build_step_parts(step_dir: Path) -> list[dict]:
    parts: list[dict] = []

    prompt_path = step_dir / "prompt.txt"
    if prompt_path.exists():
        parts.append({"text": read_text(prompt_path)})

    transcript_path = step_dir / "audio_script.txt"
    if transcript_path.exists() and not prompt_path.exists():
        parts.append({"text": read_text(transcript_path)})

    for media_path in sorted(step_dir.iterdir(), key=lambda p: p.name):
        if not media_path.is_file():
            continue
        if media_path.name in {
            "prompt.txt",
            "audio_script.txt",
            "audio_fallback_reason.txt",
            "response.txt",
            "response.json",
            "request.json",
            "error.txt",
        }:
            continue
        if media_path.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".mp3",
            ".wav",
            ".ogg",
        }:
            parts.append(inline_data_part(media_path))

    return parts


def build_combined_parts(scenario_dir: Path) -> list[dict]:
    parts: list[dict] = []

    for i, step_name in enumerate(STEPS, start=1):
        step_dir = scenario_dir / step_name
        if not step_dir.exists():
            raise FileNotFoundError(f"Missing step folder: {step_dir}")

        parts.append(
            {
                "text": (
                    f"Step {i} context ({step_name}). "
                    "Use this together with other provided assets."
                )
            }
        )
        step_parts = build_step_parts(step_dir)
        parts.extend(step_parts)

    if not parts:
        raise ValueError(f"No prompt assets found in {scenario_dir}")

    return parts


def target_needs_processing(
    responses_root: Path, index_name: str, scenario_name: str
) -> bool:
    out_dir = responses_root / index_name / scenario_name
    if not out_dir.exists():
        return True
    if (out_dir / "error.txt").exists():
        return True
    if not is_nonempty_text_file(out_dir / "response.txt"):
        return True
    return not (out_dir / "response.json").exists()


def build_inlined_requests(
    *,
    prompts_root: Path,
    responses_root: Path,
    model: str,
    temperature: float,
    max_output_tokens: int,
    reprocess: bool,
) -> tuple[list[types.InlinedRequest], list[dict]]:
    requests: list[types.InlinedRequest] = []
    manifest: list[dict] = []

    for index_dir in list_index_dirs(prompts_root):
        index_name = index_dir.name

        for scenario_name in ["scenario_1", "scenario_2"]:
            scenario_dir = index_dir / scenario_name
            if not scenario_dir.exists():
                continue

            if (not reprocess) and (
                not target_needs_processing(responses_root, index_name, scenario_name)
            ):
                continue

            combined_parts = build_combined_parts(scenario_dir)
            metadata = {
                "index": index_name,
                "scenario": scenario_name,
                "kind": "one_step_combined",
            }
            config = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_p": 1,
            }
            contents = [{"role": "user", "parts": combined_parts}]

            requests.append(
                types.InlinedRequest(
                    model=model,
                    contents=contents,
                    metadata=metadata,
                    config=config,
                )
            )
            manifest.append(metadata)

    return requests, manifest


def jobs_dir(responses_root: Path) -> Path:
    return responses_root / "_batch_jobs"


def job_state_path(responses_root: Path) -> Path:
    return jobs_dir(responses_root) / "one_step_combined.json"


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
        return False, f"one-step job still running: {state_name} ({job_name})"

    requests_manifest = job_state.get("requests", [])
    responses = []
    if job.dest and getattr(job.dest, "inlined_responses", None):
        responses = list(job.dest.inlined_responses)

    if state_name == "JOB_STATE_SUCCEEDED" and not responses:
        return (
            True,
            f"one-step job succeeded but returned no inlined responses ({job_name})",
        )

    for i, item in enumerate(responses):
        metadata = getattr(item, "metadata", None) or {}
        if not metadata and i < len(requests_manifest):
            metadata = requests_manifest[i]

        index_name = str(metadata.get("index", ""))
        scenario_name = str(metadata.get("scenario", ""))
        if not index_name or not scenario_name:
            continue

        out_dir = responses_root / index_name / scenario_name
        ensure_dir(out_dir)

        item_error = getattr(item, "error", None)
        if item_error:
            write_text(out_dir / "error.txt", str(item_error))
            continue

        response_obj = getattr(item, "response", None)
        text = extract_response_text(response_obj)
        if not text:
            write_text(
                out_dir / "error.txt",
                f"No text in batch response for {index_name}/{scenario_name}",
            )
            continue

        clear_error_if_present(out_dir)
        write_text(out_dir / "response.txt", text)
        if response_obj is not None and hasattr(response_obj, "model_dump"):
            write_json(out_dir / "response.json", response_obj.model_dump())

    job_state["state"] = state_name
    job_state["collected_at"] = now_iso()
    set_job_state(responses_root, job_state)
    return True, f"Collected one-step job: {job_name} ({state_name})"


def any_non_terminal_job(responses_root: Path) -> bool:
    state = get_job_state(responses_root)
    if not state:
        return False
    current = str(state.get("state", ""))
    return current not in TERMINAL_STATES


def submit_batch(
    *,
    client: genai.Client,
    prompts_root: Path,
    responses_root: Path,
    model: str,
    temperature: float,
    max_output_tokens: int,
    reprocess: bool,
) -> None:
    if any_non_terminal_job(responses_root):
        print("Active non-terminal one-step job already exists. Run again later.")
        return

    requests, manifest = build_inlined_requests(
        prompts_root=prompts_root,
        responses_root=responses_root,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        reprocess=reprocess,
    )

    if not requests:
        print("No requests to submit for one-step combined prompts.")
        return

    print(f"Submitting one-step combined batch: {len(requests)} requests")
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

    print(f"Submitted one-step job: {job_name} ({state_name})")


def main() -> None:
    args = parse_args()
    api_key = API_KEY
    if not api_key:
        raise ValueError("Missing Gemini API key. Set GEMINI_API_KEY.")

    prompts_root = Path(args.prompts_root)
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
            prompts_root=prompts_root,
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
