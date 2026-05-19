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
PROMPTS_ROOT = Path("experiment/generated/prompts/text_only_multistep")
RESPONSES_ROOT = Path("experiment/generated/responses/text_only_multistep")
GEMINI_MODEL = "gemini-3.1-pro-preview"
SCENARIOS = ["scenario_1", "scenario_2"]
STEPS = ["step_0", "step_1", "step_2"]
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_RETRIES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send text-only prompt requests to Gemini one-by-one and write results under "
            "text-only responses. Can resume a scenario from the first broken step."
        )
    )
    parser.add_argument(
        "--prompts-root",
        default=str(PROMPTS_ROOT),
        help="Root folder that contains text-only prompt assets by index, scenario, and step.",
    )
    parser.add_argument(
        "--responses-root",
        default=str(RESPONSES_ROOT),
        help="Root folder where text-only responses outputs are written.",
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
        help=(
            "Process only scenarios that have a missing/error step, starting from the "
            "first broken step and continuing through the remaining steps."
        ),
    )
    parser.add_argument(
        "--reprocess-all",
        action="store_true",
        help="Ignore existing responses and regenerate every selected step.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def is_nonempty_text_file(path: Path) -> bool:
    return path.exists() and path.is_file() and bool(load_text(path).strip())


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
        parts.append({"text": load_text(prompt_path).strip()})
        handled_text_files.add("prompt.txt")

    transcript_path = step_dir / "audio_script.txt"
    if (
        is_nonempty_text_file(transcript_path)
        and "prompt.txt" not in handled_text_files
    ):
        parts.append({"text": load_text(transcript_path).strip()})
        handled_text_files.add("audio_script.txt")

    ignored_generated = {
        "response.txt",
        "response.json",
        "request.json",
        "conversation.json",
        "error.txt",
        "audio_fallback_reason.txt",
    }

    for text_path in sorted(step_dir.glob("*.txt"), key=lambda p: p.name):
        if text_path.name in handled_text_files or text_path.name in ignored_generated:
            continue
        text_value = load_text(text_path).strip()
        if text_value:
            parts.append({"text": f"{text_path.name}:\n{text_value}"})

    for media_path in sorted(step_dir.iterdir(), key=lambda p: p.name):
        if not media_path.is_file():
            continue
        if media_path.name in {"prompt.txt", "audio_script.txt", "audio_fallback_reason.txt"}:
            continue
        if media_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".mp3", ".wav", ".ogg"}:
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

    ensure_dir(output_dir)
    write_json(output_dir / "request.json", {"contents": request_contents})

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


def step_is_complete(step_dir: Path) -> bool:
    if (step_dir / "error.txt").exists():
        return False
    return (
        is_nonempty_text_file(step_dir / "response.txt")
        and (step_dir / "response.json").exists()
        and (step_dir / "conversation.json").exists()
    )


def load_history_until(
    *,
    prompts_root: Path,
    responses_root: Path,
    index_name: str,
    scenario_name: str,
    end_exclusive_step_idx: int,
) -> list[dict]:
    history: list[dict] = []
    for step_name in STEPS[:end_exclusive_step_idx]:
        prompt_step_dir = prompts_root / index_name / scenario_name / step_name
        response_step_dir = responses_root / index_name / scenario_name / step_name
        response_path = response_step_dir / "response.txt"
        if not is_nonempty_text_file(response_path):
            raise FileNotFoundError(
                f"Missing prerequisite response for {index_name}/{scenario_name}/{step_name}"
            )

        user_parts = build_user_parts(prompt_step_dir)
        model_text = load_text(response_path)
        history.append({"role": "user", "parts": user_parts})
        history.append({"role": "model", "parts": [{"text": model_text}]})

    return history


def find_first_broken_step(output_scenario_dir: Path) -> str | None:
    for step_name in STEPS:
        if not step_is_complete(output_scenario_dir / step_name):
            return step_name
    return None


def selected(values: list[str], candidate: str) -> bool:
    return not values or candidate in values


def process_scenario(
    *,
    api_key: str,
    model: str,
    prompts_root: Path,
    responses_root: Path,
    index_name: str,
    scenario_name: str,
    temperature: float,
    max_output_tokens: int,
    retries: int,
    repair_broken: bool,
    reprocess_all: bool,
) -> str:
    prompt_scenario_dir = prompts_root / index_name / scenario_name
    output_scenario_dir = responses_root / index_name / scenario_name
    ensure_dir(output_scenario_dir)

    if reprocess_all:
        start_step = STEPS[0]
    elif repair_broken:
        start_step = find_first_broken_step(output_scenario_dir)
        if start_step is None:
            print(f"[{index_name}/{scenario_name}] SKIP (already complete)")
            return "skipped"
    else:
        start_step = STEPS[0]
        if find_first_broken_step(output_scenario_dir) is None:
            print(f"[{index_name}/{scenario_name}] SKIP (already complete)")
            return "skipped"

    start_idx = STEPS.index(start_step)
    history = load_history_until(
        prompts_root=prompts_root,
        responses_root=responses_root,
        index_name=index_name,
        scenario_name=scenario_name,
        end_exclusive_step_idx=start_idx,
    )

    for step_name in STEPS[start_idx:]:
        step_dir = prompt_scenario_dir / step_name
        if not step_dir.exists():
            raise FileNotFoundError(f"Missing step folder: {step_dir}")

        out_step_dir = output_scenario_dir / step_name
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
        error_file = out_step_dir / "error.txt"
        if error_file.exists():
            error_file.unlink()
        print(f"[{index_name}/{scenario_name}/{step_name}] OK")

    return "processed"


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
            if not (index_dir / scenario_name).exists():
                continue

            try:
                status = process_scenario(
                    api_key=api_key,
                    model=args.model,
                    prompts_root=prompts_root,
                    responses_root=responses_root,
                    index_name=index_name,
                    scenario_name=scenario_name,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    retries=args.retries,
                    repair_broken=args.repair_broken,
                    reprocess_all=args.reprocess_all,
                )
                if status == "processed":
                    processed += 1
            except Exception as err:
                first_broken = find_first_broken_step(
                    responses_root / index_name / scenario_name
                )
                target_step = first_broken or STEPS[0]
                out_step_dir = responses_root / index_name / scenario_name / target_step
                ensure_dir(out_step_dir)
                write_text(out_step_dir / "error.txt", str(err))
                print(f"[{index_name}/{scenario_name}] FAILED: {err}")
                failed += 1

    print(f"Done. Processed: {processed}, Failed: {failed}")


if __name__ == "__main__":
    main()
