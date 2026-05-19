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
RESPONSES_ROOT = Path("experiment/generated/responses/multistep_multimodal")
GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_RETRIES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send step prompts to Gemini while preserving scenario chat context."
    )
    parser.add_argument(
        "--prompts-root",
        default=str(PROMPTS_ROOT),
        help="Root folder that contains index/scenario/step prompt assets.",
    )
    parser.add_argument(
        "--responses-root",
        default=str(RESPONSES_ROOT),
        help="Root folder where responses are written.",
    )
    parser.add_argument(
        "--model",
        default=GEMINI_MODEL,
        help="Gemini model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum output tokens per step.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Retries per request.",
    )
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Process only indices that currently have responses-root/index/error.txt.",
    )
    parser.add_argument(
        "--reprocess-all",
        action="store_true",
        help="Ignore existing responses and process every index.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_index_dirs(prompts_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(prompts_root.iterdir(), key=lambda p: p.name)
        if path.is_dir()
    ]


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def file_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix == ".jpg" or suffix == ".jpeg":
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

    prompt_path = step_dir / "prompt.txt"
    if prompt_path.exists():
        parts.append({"text": load_text(prompt_path)})

    transcript_path = step_dir / "audio_script.txt"
    if transcript_path.exists() and not prompt_path.exists():
        parts.append({"text": load_text(transcript_path)})

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
            "conversation.json",
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

    if not parts:
        raise ValueError(f"No prompt assets found in {step_dir}")

    return parts


def build_request_contents(history: list[dict], user_parts: list[dict]) -> list[dict]:
    return [*history, {"role": "user", "parts": user_parts}]


def extract_response_text(response_json: dict) -> str:
    candidates = response_json.get("candidates", [])
    if not candidates:
        raise ValueError(f"Gemini response missing candidates: {response_json}")

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    text_parts = [str(part.get("text", "")) for part in parts if isinstance(part, dict)]
    text = "".join(text_parts).strip()
    if not text:
        raise ValueError(f"Gemini response did not contain text: {response_json}")
    return text


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
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
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


def run_step(
    *,
    api_key: str,
    model: str,
    history: list[dict],
    step_dir: Path,
    output_dir: Path,
    temperature: float,
    max_output_tokens: int,
    retries: int,
) -> tuple[list[dict], dict, str]:
    user_parts = build_user_parts(step_dir)
    request_contents = build_request_contents(history, user_parts)

    step_output_dir = output_dir
    ensure_dir(step_output_dir)
    write_json(step_output_dir / "request.json", {"contents": request_contents})

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            response_json = call_gemini(
                api_key=api_key,
                model=model,
                contents=request_contents,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            response_text = extract_response_text(response_json)

            new_history = [
                *history,
                {"role": "user", "parts": user_parts},
                {"role": "model", "parts": [{"text": response_text}]},
            ]
            return new_history, response_json, response_text
        except Exception as err:
            last_err = err
            if attempt == retries:
                break
            err_text = str(err)
            if "429" in err_text or "503" in err_text:
                time.sleep(min(10 * attempt, 60))
            else:
                time.sleep(min(2**attempt, 10))

    raise RuntimeError(f"Gemini request failed after {retries} attempts: {last_err}")


def process_scenario(
    *,
    api_key: str,
    model: str,
    scenario_dir: Path,
    output_scenario_dir: Path,
    temperature: float,
    max_output_tokens: int,
    retries: int,
) -> None:
    history: list[dict] = []

    for step_name in ["step_0", "step_1", "step_2"]:
        step_dir = scenario_dir / step_name
        if not step_dir.exists():
            raise FileNotFoundError(f"Missing step folder: {step_dir}")

        out_step_dir = output_scenario_dir / step_name
        ensure_dir(out_step_dir)

        history, response_json, response_text = run_step(
            api_key=api_key,
            model=model,
            history=history,
            step_dir=step_dir,
            output_dir=out_step_dir,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            retries=retries,
        )

        write_text(out_step_dir / "response.txt", response_text)
        write_json(out_step_dir / "response.json", response_json)
        write_json(out_step_dir / "conversation.json", history)


def index_is_complete(output_index_dir: Path) -> bool:
    if (output_index_dir / "error.txt").exists():
        return False

    for scenario_name in ["scenario_1", "scenario_2"]:
        for step_name in ["step_0", "step_1", "step_2"]:
            step_dir = output_index_dir / scenario_name / step_name
            if not step_dir.exists():
                return False
            if not (step_dir / "response.txt").exists():
                return False
            if not (step_dir / "response.json").exists():
                return False
            if not (step_dir / "conversation.json").exists():
                return False

    return True


def main() -> None:
    args = parse_args()
    api_key = API_KEY
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY environment variable.")

    prompts_root = Path(args.prompts_root)
    responses_root = Path(args.responses_root)
    ensure_dir(responses_root)

    failed_indices = set()
    if args.only_failed:
        for index_dir in responses_root.iterdir():
            if index_dir.is_dir() and (index_dir / "error.txt").exists():
                failed_indices.add(index_dir.name)

        if not failed_indices:
            print("No previously failed indices found.")
            print("Done. Processed: 0, Failed: 0")
            return

    processed = 0
    failed = 0

    for index_dir in list_index_dirs(prompts_root):
        if args.only_failed and index_dir.name not in failed_indices:
            continue

        output_index_dir = responses_root / index_dir.name
        if not args.reprocess_all and not args.only_failed and index_is_complete(output_index_dir):
            print(f"[{index_dir.name}] SKIP (already complete)")
            continue

        try:
            ensure_dir(output_index_dir)
            for scenario_name in ["scenario_1", "scenario_2"]:
                scenario_dir = index_dir / scenario_name
                if not scenario_dir.exists():
                    raise FileNotFoundError(f"Missing scenario folder: {scenario_dir}")
                output_scenario_dir = output_index_dir / scenario_name
                ensure_dir(output_scenario_dir)
                process_scenario(
                    api_key=api_key,
                    model=args.model,
                    scenario_dir=scenario_dir,
                    output_scenario_dir=output_scenario_dir,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    retries=args.retries,
                )

            error_file = output_index_dir / "error.txt"
            if error_file.exists():
                error_file.unlink()

            print(f"[{index_dir.name}] OK")
            processed += 1
        except Exception as err:
            write_text(output_index_dir / "error.txt", str(err))
            print(f"[{index_dir.name}] FAILED: {err}")
            failed += 1

    print(f"Done. Processed: {processed}, Failed: {failed}")


if __name__ == "__main__":
    main()
