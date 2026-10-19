# LLM-assisted cleaning of report-derived labels in CT-RATE

Code for the study "Large Language Model–Assisted Cleaning of Report-Derived Labels in a Large-Scale Chest CT Dataset".
Data, refined labels, radiologist annotations, and run logs will be released separately.

## Contents

| Path | Purpose |
|---|---|
| `src/extract_labels.py` | Full-dataset label extraction (binary output) with GPT-5.4, GPT-5.4 mini, and DeepSeek V3.2 via Azure AI Foundry |
| `src/extract_labels_4level.py` | Four-level output variant (present / uncertain / negated / not mentioned) with the same label definitions |
| `src/repeat_runs.py` | Repeat-run experiment on the confirmatory reference set, with per-request logs (timestamps, attempts) |
| `src/sample_confirmatory_set.py` | Stratified sampling of the 250-report confirmatory reference set |
| `src/evaluate_confirmatory.py` | Per-category and label-macro/micro performance, unweighted and design-weighted, report-level bootstrap |
| `src/paired_confirmatory.py` | Exact McNemar-type tests, paired bootstrap differences, Holm correction |
| `src/sensitivity_rules.py` | Binarization rules A–D |
| `src/intermodel_confirmatory.py` | Pairwise agreement, error correlation, leave-one-model-out |
| `src/analyze_repeat_runs.py` | Run-to-run flip rate and Fleiss κ |
| `src/evaluate_4level.py` | Evaluation of four-level LLM outputs |
| `src/build_annotation_tool.py`, `src/autofill_from_history.py` | Radiologist annotation tool |
| `notebooks/` | Analyses for the full dataset and the development reference set (outputs cleared) |

## Environment

Python version and dependencies are pinned in `.python-version`, `pyproject.toml`, and `uv.lock` (`uv sync`).
Credentials are read from `AZURE_AI_ENDPOINT` and `AZURE_AI_KEY`; no credentials are stored in this repository.
