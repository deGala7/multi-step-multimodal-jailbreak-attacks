#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from json_repair import repair_json


GROK_DIR = Path("experiment/generated/decomposition/harmful")
OUTPUT_ROOT = Path("experiment/generated/prompts/text_only_multistep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate text-only prompt assets from decomposition response_raw files. "
            "Keeps the same scenario/step folder structure as the multimodal prompt assets."
        )
    )
    parser.add_argument(
        "--grok-dir",
        default=str(GROK_DIR),
        help="Root folder containing decomposition response_raw.txt files.",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help="Output folder for text-only prompt assets.",
    )
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Process only indices that currently have error.txt under the text-only prompt output folder.",
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


def parse_model_json(maybe_json: str) -> dict:
    text = extract_json_text(maybe_json)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    repaired = repair_json(text, return_objects=True)
    if isinstance(repaired, dict):
        return repaired

    raise ValueError("Response is not a valid JSON object after repair.")


def load_response_json(response_raw_path: Path) -> dict:
    raw = response_raw_path.read_text(encoding="utf-8")
    try:
        return parse_model_json(raw)
    except Exception as raw_err:
        api_response_path = response_raw_path.with_name("api_response.json")
        if api_response_path.exists():
            api_json = json.loads(api_response_path.read_text(encoding="utf-8"))
            content = (
                api_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            if content:
                return parse_model_json(content)
        raise raw_err


def build_mapping_table_text(mapping_table: list[dict]) -> str:
    lines = ["ID | ACTUAL_MEANING | STATUS", "-" * 72]
    for row in mapping_table:
        id_value = str(row.get("id", "")).strip()
        meaning = str(row.get("actual_meaning", "")).strip()
        status = str(row.get("status", "")).strip()
        lines.append(f"{id_value} | {meaning} | {status}")
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def generate_for_index(index_dir: Path, output_root: Path) -> None:
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

    write_text(out_index / "target_summary.txt", str(data.get("target_summary", "")))

    write_text(s1_step0 / "prompt.txt", str(scenario_1.get("audio_persona_script", "")))
    write_text(s1_step1 / "image_a.txt", str(scenario_1.get("image_a_content", "")))
    write_text(s1_step1 / "image_b.txt", str(scenario_1.get("image_b_content", "")))
    write_text(
        s1_step2 / "prompt.txt", str(scenario_1.get("final_synthesis_prompt", ""))
    )

    write_text(s2_step0 / "prompt.txt", str(scenario_2.get("audio_audit_script", "")))
    write_text(
        s2_step1 / "mapping_table.txt",
        build_mapping_table_text(scenario_2.get("mapping_table", [])),
    )
    write_text(
        s2_step2 / "prompt.txt", str(scenario_2.get("final_technical_prompt", ""))
    )


def main() -> None:
    args = parse_args()
    grok_dir = Path(args.grok_dir)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    processed = 0
    failed = 0

    failed_indices: set[str] = set()

    if args.only_failed:
        all_indices = {
            index_dir.name for index_dir in grok_dir.iterdir() if index_dir.is_dir()
        }

        for index_name in all_indices:
            out_index_dir = output_root / index_name
            if not out_index_dir.exists() or (out_index_dir / "error.txt").exists():
                failed_indices.add(index_name)

    for index_dir in sorted(grok_dir.iterdir(), key=lambda p: p.name):
        if not index_dir.is_dir():
            continue
        if args.only_failed and index_dir.name not in failed_indices:
            continue

        try:
            generate_for_index(index_dir, output_root)
            error_file = output_root / index_dir.name / "error.txt"
            if error_file.exists():
                error_file.unlink()
            print(f"[{index_dir.name}] OK")
            processed += 1
        except Exception as err:
            error_dir = output_root / index_dir.name
            ensure_dir(error_dir)
            write_text(error_dir / "error.txt", str(err))
            print(f"[{index_dir.name}] FAILED: {err}")
            failed += 1

    print(f"Done. Processed: {processed}, Failed: {failed}")


if __name__ == "__main__":
    main()
