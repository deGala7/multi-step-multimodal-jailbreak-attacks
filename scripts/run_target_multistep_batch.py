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
RESPONSES_ROOT = Path("experiment/generated/responses/multistep_multimodal")
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
            "Run Gemini Batch API for one step at a time. "
            "On each run it first collects completed batch jobs, then submits "
            "a new job for missing/error outputs of the requested step."
        )
    )
    parser.add_argument(
        "--step",
        choices=STEPS,
        required=True,
        help="Which step to process in this run.",
    )
    parser.add_argument(
        "--prompts-root",
        default=str(PROMPTS_ROOT),
        help="Root folder containing multimodal prompt assets by index, scenario, and step.",
    )
    parser.add_argument(
        "--responses-root",
        default=str(RESPONSES_ROOT),
        help="Root folder for generated responses.",
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
        help="Include already completed step outputs in the new batch.",
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


def build_user_parts(step_dir: Path) -> list[dict]:
    parts: list[dict] = []
    handled_text_files: set[str] = set()

    prompt_path = step_dir / "prompt.txt"
    if is_nonempty_text_file(prompt_path):
        parts.append({"text": read_text(prompt_path).strip()})
        handled_text_files.add("prompt.txt")

    transcript_path = step_dir / "audio_script.txt"
    if (
        is_nonempty_text_file(transcript_path)
        and "prompt.txt" not in handled_text_files
    ):
        parts.append({"text": read_text(transcript_path).strip()})
        handled_text_files.add("audio_script.txt")

    ignored_generated = {
        "response.txt",
        "response.json",
        "request.json",
        "error.txt",
        "audio_fallback_reason.txt",
    }

    # Include any extra text prompt files beyond the standard prompt/audio script names.
    for text_path in sorted(step_dir.glob("*.txt"), key=lambda p: p.name):
        if text_path.name in handled_text_files or text_path.name in ignored_generated:
            continue
        text_value = read_text(text_path).strip()
        if not text_value:
            continue
        parts.append({"text": f"{text_path.name}:\n{text_value}"})

    for media_path in sorted(step_dir.iterdir(), key=lambda p: p.name):
        if not media_path.is_file():
            continue
        if media_path.name in {
            "prompt.txt",
            "audio_script.txt",
            "audio_fallback_reason.txt",
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

    if not parts:
        raise ValueError(f"No prompt assets found in {step_dir}")

    return parts


def step_index(step_name: str) -> int:
    return STEPS.index(step_name)


def load_previous_history(
    *,
    prompts_root: Path,
    responses_root: Path,
    index_name: str,
    scenario_name: str,
    target_step: str,
) -> list[dict]:
    history: list[dict] = []
    target_idx = step_index(target_step)

    for prev_step in STEPS[:target_idx]:
        step_dir = responses_root / index_name / scenario_name / prev_step
        response_path = step_dir / "response.txt"
        if not is_nonempty_text_file(response_path):
            raise FileNotFoundError(
                f"Missing prerequisite response for {index_name}/{scenario_name}/{prev_step}"
            )

        prompt_step_dir = prompts_root / index_name / scenario_name / prev_step
        user_parts = build_user_parts(prompt_step_dir)
        model_text = read_text(response_path)

        history.append({"role": "user", "parts": user_parts})
        history.append({"role": "model", "parts": [{"text": model_text}]})

    return history


def target_step_needs_processing(
    responses_root: Path, index_name: str, scenario_name: str, step_name: str
) -> bool:
    step_dir = responses_root / index_name / scenario_name / step_name
    if not step_dir.exists():
        return True
    if (step_dir / "error.txt").exists():
        return True
    if not is_nonempty_text_file(step_dir / "response.txt"):
        return True
    return not (step_dir / "response.json").exists()


def build_inlined_requests(
    *,
    prompts_root: Path,
    responses_root: Path,
    step_name: str,
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
                not target_step_needs_processing(
                    responses_root, index_name, scenario_name, step_name
                )
            ):
                continue

            step_dir = scenario_dir / step_name
            if not step_dir.exists():
                continue

            try:
                history = load_previous_history(
                    prompts_root=prompts_root,
                    responses_root=responses_root,
                    index_name=index_name,
                    scenario_name=scenario_name,
                    target_step=step_name,
                )
            except FileNotFoundError:
                # Previous-step context is not ready yet for this scenario.
                continue

            user_parts = build_user_parts(step_dir)
            contents = [*history, {"role": "user", "parts": user_parts}]

            metadata = {
                "index": index_name,
                "scenario": scenario_name,
                "step": step_name,
            }
            config = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_p": 1,
            }

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


def job_state_path(responses_root: Path, step_name: str) -> Path:
    return jobs_dir(responses_root) / f"{step_name}.json"


def get_job_state(responses_root: Path, step_name: str) -> dict | None:
    path = job_state_path(responses_root, step_name)
    if not path.exists():
        return None
    return load_json(path)


def set_job_state(responses_root: Path, step_name: str, state: dict) -> None:
    ensure_dir(jobs_dir(responses_root))
    write_json(job_state_path(responses_root, step_name), state)


def clear_error_if_present(step_output_dir: Path) -> None:
    error_path = step_output_dir / "error.txt"
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
    step_name: str,
    job_state: dict,
) -> tuple[bool, str]:
    job_name = job_state.get("job_name", "")
    if not job_name:
        return True, "Invalid job state (missing job_name)."

    job = client.batches.get(name=job_name)
    state_name = getattr(job.state, "name", str(job.state))

    if state_name not in TERMINAL_STATES:
        return False, f"{step_name} job still running: {state_name} ({job_name})"

    requests_manifest = job_state.get("requests", [])
    responses = []
    if job.dest and getattr(job.dest, "inlined_responses", None):
        responses = list(job.dest.inlined_responses)

    if state_name == "JOB_STATE_SUCCEEDED" and not responses:
        return (
            True,
            f"{step_name} job succeeded but returned no inlined responses ({job_name})",
        )

    for i, item in enumerate(responses):
        metadata = getattr(item, "metadata", None) or {}
        if not metadata and i < len(requests_manifest):
            metadata = requests_manifest[i]

        index_name = str(metadata.get("index", ""))
        scenario_name = str(metadata.get("scenario", ""))
        if not index_name or not scenario_name:
            continue

        out_step_dir = responses_root / index_name / scenario_name / step_name
        ensure_dir(out_step_dir)

        item_error = getattr(item, "error", None)
        if item_error:
            write_text(out_step_dir / "error.txt", str(item_error))
            continue

        response_obj = getattr(item, "response", None)
        text = extract_response_text(response_obj)
        if not text:
            write_text(
                out_step_dir / "error.txt",
                f"No text in batch response for {index_name}/{scenario_name}/{step_name}",
            )
            continue

        clear_error_if_present(out_step_dir)
        write_text(out_step_dir / "response.txt", text)
        if response_obj is not None and hasattr(response_obj, "model_dump"):
            write_json(out_step_dir / "response.json", response_obj.model_dump())

    job_state["state"] = state_name
    job_state["collected_at"] = now_iso()
    set_job_state(responses_root, step_name, job_state)
    return True, f"Collected {step_name} job: {job_name} ({state_name})"


def refresh_all_known_jobs(client: genai.Client, responses_root: Path) -> None:
    for step_name in STEPS:
        state = get_job_state(responses_root, step_name)
        if not state:
            continue
        finished, message = collect_job_results(
            client=client,
            responses_root=responses_root,
            step_name=step_name,
            job_state=state,
        )
        print(message)
        if not finished:
            continue


def any_non_terminal_job(responses_root: Path, step_name: str) -> bool:
    state = get_job_state(responses_root, step_name)
    if not state:
        return False
    current = str(state.get("state", ""))
    return current not in TERMINAL_STATES


def submit_batch_for_step(
    *,
    client: genai.Client,
    prompts_root: Path,
    responses_root: Path,
    step_name: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    reprocess: bool,
) -> None:
    if any_non_terminal_job(responses_root, step_name):
        print(
            f"Active non-terminal job already exists for {step_name}. Run again later."
        )
        return

    requests, manifest = build_inlined_requests(
        prompts_root=prompts_root,
        responses_root=responses_root,
        step_name=step_name,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        reprocess=reprocess,
    )

    if not requests:
        print(f"No requests to submit for {step_name}.")
        return

    print(f"Submitting batch for {step_name}: {len(requests)} requests")
    batch_job = client.batches.create(model=model, src=requests)
    job_name = str(batch_job.name)
    state_name = getattr(batch_job.state, "name", str(batch_job.state))

    set_job_state(
        responses_root,
        step_name,
        {
            "step": step_name,
            "job_name": job_name,
            "state": state_name,
            "submitted_at": now_iso(),
            "request_count": len(requests),
            "requests": manifest,
        },
    )

    print(f"Submitted {step_name} job: {job_name} ({state_name})")


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

    # First pass: collect any finished jobs (including previous steps).
    refresh_all_known_jobs(client, responses_root)

    # Then submit a new job for the requested step unless collect-only is requested.
    if not args.collect_only:
        submit_batch_for_step(
            client=client,
            prompts_root=prompts_root,
            responses_root=responses_root,
            step_name=args.step,
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            reprocess=args.reprocess,
        )
    else:
        print("Collection pass done (collect-only mode).")


if __name__ == "__main__":
    main()
