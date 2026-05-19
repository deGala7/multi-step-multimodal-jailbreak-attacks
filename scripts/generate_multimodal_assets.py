#!/usr/bin/env python3

import argparse
import asyncio
import json
import textwrap
import time
import wave
from pathlib import Path

import edge_tts
from gtts import gTTS
from json_repair import repair_json
from PIL import Image, ImageDraw, ImageFont


GROK_DIR = Path("experiment/generated/decomposition/harmful")
OUTPUT_ROOT = Path("experiment/generated/prompts/multistep_multimodal")
IMAGE_SIZE = (1600, 1200)
MARGIN = 60
LINE_WIDTH = 80
TTS_RETRIES = 5
TTS_PROVIDER = "edge"
EDGE_TTS_VOICE = "en-US-AriaNeural"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate audio/image/text prompt assets from decomposition outputs."
    )
    parser.add_argument(
        "--grok-dir",
        default=str(GROK_DIR),
        help="Root folder containing decomposition response_raw.txt files.",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help="Output folder for multimodal prompt assets.",
    )
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Process only indices that currently have error.txt under the output folder.",
    )
    parser.add_argument(
        "--tts-provider",
        choices=["edge", "gtts"],
        default=TTS_PROVIDER,
        help="Text-to-speech provider to use for audio assets.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_model_json(maybe_json: str) -> dict:
    text = extract_json_text(maybe_json)

    # First, try strict JSON parsing.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fall back to automatic repair for malformed escapes/control characters.
    repaired = repair_json(text, return_objects=True)
    if isinstance(repaired, dict):
        return repaired

    raise ValueError("Response is not a valid JSON object after repair.")


def load_response_json(response_raw_path: Path) -> dict:
    raw = response_raw_path.read_text(encoding="utf-8")
    try:
        return _parse_model_json(raw)
    except Exception as raw_err:
        # Some responses are malformed only in response_raw.txt.
        # In those cases, fallback to the same content from api_response.json.
        api_response_path = response_raw_path.with_name("api_response.json")
        if api_response_path.exists():
            api_json = json.loads(api_response_path.read_text(encoding="utf-8"))
            content = (
                api_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            if content:
                return _parse_model_json(content)
        raise raw_err


def wrap_text_blocks(text: str, width: int = LINE_WIDTH) -> str:
    out_lines = []
    for block in text.split("\n"):
        if not block.strip():
            out_lines.append("")
            continue
        out_lines.extend(textwrap.wrap(block, width=width, break_long_words=False))
    return "\n".join(out_lines)


def write_text_image(text: str, output_path: Path) -> None:
    image = Image.new("RGB", IMAGE_SIZE, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    wrapped = wrap_text_blocks(text)
    draw.multiline_text((MARGIN, MARGIN), wrapped, fill=(0, 0, 0), font=font, spacing=8)
    image.save(output_path)


def build_mapping_table_text(mapping_table: list[dict]) -> str:
    lines = ["ID | ACTUAL_MEANING | STATUS", "-" * 72]
    for row in mapping_table:
        id_value = str(row.get("id", "")).strip()
        meaning = str(row.get("actual_meaning", "")).strip()
        status = str(row.get("status", "")).strip()
        lines.append(f"{id_value} | {meaning} | {status}")
    return "\n".join(lines)


async def _write_audio_edge(script_text: str, output_path: Path) -> None:
    communicate = edge_tts.Communicate(script_text, EDGE_TTS_VOICE)
    await communicate.save(str(output_path))


def write_audio(script_text: str, output_path: Path, provider: str) -> None:
    last_err = None
    for attempt in range(1, TTS_RETRIES + 1):
        try:
            if provider == "edge":
                asyncio.run(_write_audio_edge(script_text, output_path))
            else:
                tts = gTTS(text=script_text, lang="en")
                tts.save(str(output_path))
            return
        except Exception as err:
            last_err = err
            if attempt == TTS_RETRIES:
                break
            err_text = str(err)
            # gTTS can rate-limit bursts; use stronger backoff on HTTP 429.
            if "429" in err_text:
                time.sleep(min(20 * attempt, 120))
            else:
                time.sleep(min(2**attempt, 20))

    # Fallback: emit a valid local WAV so generation can continue.
    fallback_wav = output_path.with_suffix(".wav")
    with wave.open(str(fallback_wav), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000)  # 1 second silence

    output_path.with_name("audio_script.txt").write_text(script_text, encoding="utf-8")
    output_path.with_name("audio_fallback_reason.txt").write_text(
        f"gTTS failed after {TTS_RETRIES} attempts: {last_err}\n"
        f"Generated fallback audio file: {fallback_wav.name}\n",
        encoding="utf-8",
    )


def generate_for_index(
    index_dir: Path, output_root: Path, provider: str = TTS_PROVIDER
) -> None:
    generate_for_index_with_provider(index_dir, output_root, provider)


def generate_for_index_with_provider(
    index_dir: Path, output_root: Path, provider: str
) -> None:
    response_raw_path = index_dir / "response_raw.txt"
    if not response_raw_path.exists():
        return

    data = load_response_json(response_raw_path)

    scenario_1 = data.get("scenario_1", {})
    scenario_2 = data.get("scenario_2", {})

    out_index = output_root / index_dir.name

    s1_step0 = out_index / "scenario_1" / "step_0"
    s1_step1 = out_index / "scenario_1" / "step_1"
    s1_step2 = out_index / "scenario_1" / "step_2"

    s2_step0 = out_index / "scenario_2" / "step_0"
    s2_step1 = out_index / "scenario_2" / "step_1"
    s2_step2 = out_index / "scenario_2" / "step_2"

    for path in (s1_step0, s1_step1, s1_step2, s2_step0, s2_step1, s2_step2):
        ensure_dir(path)

    write_audio(
        str(scenario_1.get("audio_persona_script", "")),
        s1_step0 / "audio.mp3",
        provider,
    )
    write_text_image(
        str(scenario_1.get("image_a_content", "")),
        s1_step1 / "image_a.png",
    )
    write_text_image(
        str(scenario_1.get("image_b_content", "")),
        s1_step1 / "image_b.png",
    )
    (s1_step2 / "prompt.txt").write_text(
        str(scenario_1.get("final_synthesis_prompt", "")),
        encoding="utf-8",
    )

    write_audio(
        str(scenario_2.get("audio_audit_script", "")),
        s2_step0 / "audio.mp3",
        provider,
    )
    mapping_text = build_mapping_table_text(scenario_2.get("mapping_table", []))
    write_text_image(mapping_text, s2_step1 / "mapping_table.png")
    (s2_step2 / "prompt.txt").write_text(
        str(scenario_2.get("final_technical_prompt", "")),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    grok_dir = Path(args.grok_dir)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    processed = 0
    failed = 0

    failed_indices = set()

    if args.only_failed:
        # Collect all expected indices from the decomposition folder.
        all_indices = {
            index_dir.name for index_dir in grok_dir.iterdir() if index_dir.is_dir()
        }

        for index_name in all_indices:
            out_index_dir = output_root / index_name

            # Case 1: output folder doesn't exist → failed
            if not out_index_dir.exists():
                failed_indices.add(index_name)
                continue

            # Case 2: explicit error file → failed
            if (out_index_dir / "error.txt").exists():
                failed_indices.add(index_name)
                continue

    for index_dir in sorted(grok_dir.iterdir(), key=lambda p: p.name):
        if not index_dir.is_dir():
            continue
        if args.only_failed and index_dir.name not in failed_indices:
            continue
        try:
            generate_for_index_with_provider(index_dir, output_root, args.tts_provider)
            error_file = output_root / index_dir.name / "error.txt"
            if error_file.exists():
                error_file.unlink()
            print(f"[{index_dir.name}] OK")
            processed += 1
        except Exception as err:
            error_dir = output_root / index_dir.name
            ensure_dir(error_dir)
            (error_dir / "error.txt").write_text(str(err), encoding="utf-8")
            print(f"[{index_dir.name}] FAILED: {err}")
            failed += 1

    print(f"Done. Processed: {processed}, Failed: {failed}")


if __name__ == "__main__":
    main()
