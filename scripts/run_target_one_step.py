#!/usr/bin/env python3

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


API_KEY = os.environ.get("GEMINI_API_KEY")
PROMPTS_ROOT = Path("experiment/generated/prompts/multistep_multimodal")
RESPONSES_ROOT = Path("experiment/generated/responses/one_step_multimodal")
GEMINI_MODEL = "gemini-3.1-pro-preview"
SCENARIOS = ["scenario_1", "scenario_2"]
STEPS = ["step_0", "step_1", "step_2"]
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_RETRIES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one-step combined scenario prompts to Gemini one-by-one."
    )
    parser.add_argument(
        "--prompts-root",
        default=str(PROMPTS_ROOT),
        help="Root folder containing multimodal prompt assets by index, scenario, and step.",
    )
    parser.add_argument(
        "--responses-root",
        default=str(RESPONSES_ROOT),
        help="Root folder where one-step responses are written.",
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
        "--scenario",
        action="append",
        default=[],
        help="Specific scenario to process. Repeat to pass multiple scenarios.",
    )
    parser.add_argument(
        "--repair-broken",
        action="store_true",
        help="Process only missing/error one-step outputs.",
    )
    parser.add_argument(
        "--reprocess-all",
        action="store_true",
        help="Ignore existing responses and regenerate every selected scenario.",
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
        if media_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".mp3", ".wav", ".ogg"}:
            parts.append(inline_data_part(media_path))

    return parts


def build_combined_parts(scenario_dir: Path) -> list[dict]:
    parts: list[dict] = []
    for i, step_name in enumerate(STEPS, start=1):
        step_dir = scenario_dir / step_name
        if not step_dir.exists():
            raise FileNotFoundError(f"Missing step folder: {step_dir}")
        parts.append({"text": f"Step {i} context ({step_name}). Use this together with other provided assets."})
        parts.extend(build_step_parts(step_dir))

    if not parts:
        raise ValueError(f"No prompt assets found in {scenario_dir}")
    return parts


def scenario_is_complete(scenario_dir: Path) -> bool:
    if (scenario_dir / "error.txt").exists():
        return False
    return is_nonempty_text_file(scenario_dir / "response.txt") and (scenario_dir / "response.json").exists()


def call_gemini(
    *,
    api_key: str,
    model: str,
    contents: list[dict],
    temperature: float,
    max_output_tokens: int,
) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": contents,
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

    prompts_root = Path(args.prompts_root)
    responses_root = Path(args.responses_root)
    ensure_dir(responses_root)

    processed = 0
    failed = 0

    for index_dir in list_index_dirs(prompts_root):
        index_name = index_dir.name
        if not selected(args.index, index_name):
            continue

        for scenario_name in SCENARIOS:
            if not selected(args.scenario, scenario_name):
                continue

            scenario_dir = index_dir / scenario_name
            if not scenario_dir.exists():
                continue

            output_dir = responses_root / index_name / scenario_name
            ensure_dir(output_dir)

            if not args.reprocess_all:
                complete = scenario_is_complete(output_dir)
                if args.repair_broken and complete:
                    continue
                if not args.repair_broken and complete:
                    print(f"[{index_name}/{scenario_name}] SKIP (already complete)")
                    continue

            combined_parts = build_combined_parts(scenario_dir)
            request_contents = [{"role": "user", "parts": combined_parts}]
            write_json(output_dir / "request.json", {"contents": request_contents})

            last_err = None
            for attempt in range(1, args.retries + 1):
                try:
                    response_json = call_gemini(
                        api_key=api_key,
                        model=args.model,
                        contents=request_contents,
                        temperature=args.temperature,
                        max_output_tokens=args.max_output_tokens,
                    )
                    response_text = extract_response_text(response_json)
                    write_text(output_dir / "response.txt", response_text)
                    write_json(output_dir / "response.json", response_json)
                    error_file = output_dir / "error.txt"
                    if error_file.exists():
                        error_file.unlink()
                    print(f"[{index_name}/{scenario_name}] OK")
                    processed += 1
                    last_err = None
                    break
                except Exception as err:
                    last_err = err
                    if attempt < args.retries:
                        time.sleep(min(2**attempt, 10))

            if last_err is not None:
                write_text(output_dir / "error.txt", str(last_err))
                print(f"[{index_name}/{scenario_name}] FAILED: {last_err}")
                failed += 1

    print(f"Done. Processed: {processed}, Failed: {failed}")


if __name__ == "__main__":
    main()
