# Multi-Step Multimodal Jailbreak Attacks

This repository contains the experiment code and artifacts for a research study on multi-step multimodal jailbreak attacks against large language models.

The experiment compares direct harmful prompting with structured attack variants that split intent across turns and modalities. The goal is to make the results reproducible: prompts, generated assets, model responses, judge outputs, and statistics are stored with stable dataset indices.

## Safety Notice

This repository is for academic safety research and reproducibility. It contains harmful prompts, jailbreak attempts, model responses, and automated judge outputs. Do not use these materials to bypass safeguards or target deployed systems.

API keys are not stored in the repository. Scripts read keys from environment variables.

## What Is Included

- `experiment/data/`: harmful and benign datasets used by the scripts.
- `experiment/templates/`: prompt templates for attack decomposition and judging.
- `experiment/generated/decomposition/`: generated scenario decompositions.
- `experiment/generated/prompts/`: generated audio, image, and text prompt assets.
- `experiment/generated/responses/`: target-model responses.
- `experiment/results/judge/`: automated judge outputs.
- `experiment/results/statistics/`: computed statistics and LaTeX tables.
- `scripts/`: scripts for generation, model querying, judging, and statistics.
- `scripts/common/`: shared helpers for parsing and statistics.

Paper source, notes, and PDFs are kept outside this repository.

## Experiment Variants

- `multistep_multimodal`: three-turn attack using audio, image, and text.
- `text_only_multistep`: same three-turn structure, converted to text only.
- `one_step_multimodal`: same multimodal content sent in a single request.
- `raw_baseline`: original harmful goal sent directly as text.

The raw baseline is included to calibrate the automated judge. If raw prompting has low success while structured prompting is much higher, the result is less likely to be only a judge artifact.

## Models

- Target model: `gemini-3.1-pro-preview`
- Decomposition model: `llama-3.1-8b-instant`
- Judge model: `llama-3.1-8b-instant`

Target-model calls used temperature `0.7`. Each prompt was run once.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`, then export the variables before running scripts that call model APIs:

```bash
export GROQ_API_KEY="..."
export GEMINI_API_KEY="..."
```

## Reproduction

Generate attack decompositions:

```bash
python scripts/decompose_attacks.py
```

Generate multimodal assets and text-only prompts:

```bash
python scripts/generate_multimodal_assets.py
python scripts/generate_text_only_prompts.py
```

Run target-model queries:

```bash
python scripts/run_target_multistep.py
python scripts/run_target_text_only.py
python scripts/run_target_one_step.py
python scripts/run_target_raw.py
```

Run automated judging:

```bash
python scripts/judge_multistep.py
python scripts/judge_text_only.py
python scripts/judge_one_step.py
python scripts/judge_raw.py
```

Recompute statistics:

```bash
python scripts/compute_statistics.py
python scripts/compute_statistics_text_only.py
python scripts/compute_statistics_one_step.py
python scripts/compute_statistics_raw.py
python scripts/generate_tables.py
```

## Verification

Check that the scripts compile:

```bash
python -m py_compile scripts/*.py scripts/common/*.py
```

The main result tables are in `experiment/results/statistics/`.
