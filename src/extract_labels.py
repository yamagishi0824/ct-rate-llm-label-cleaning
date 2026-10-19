#!/usr/bin/env python3
"""extract_labels.py

Binary label assignment for CT-RATE chest CT radiology reports
using Azure AI Foundry (Responses API + Structured Outputs) or DeepSeek.

For each of the 18 CT-RATE abnormality categories, the model assigns:
  1 = finding is present / mentioned in the report
  0 = finding is absent OR not mentioned

Input : data/CT-RATE_reports.csv  (VolumeName, Findings_EN, Impressions_EN, split, ...)
Output: data/CT-RATE_reports_{model_name}.csv  (same structure, label columns overwritten)

Usage
-----
  # Azure OpenAI (GPT / o-series)
  python src/extract_labels.py \\
      --model-name o3 \\
      --endpoint  https://<resource>.services.ai.azure.com/api \\
      --api-key   <key>

  # DeepSeek via Azure AI Foundry
  python src/extract_labels.py \\
      --model-name deepseek-v3 \\
      --endpoint  https://<resource>.services.ai.azure.com/models \\
      --api-key   <key>

  # Env-var alternative
  export AZURE_AI_ENDPOINT=...
  export AZURE_AI_KEY=...
  python src/extract_labels.py --model-name o3
"""

from __future__ import annotations

import os
import re
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import AzureOpenAI, OpenAI

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # noqa: E731


# ── Label definitions ────────────────────────────────────────────────────────

LABEL_COLS: list[str] = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]

# JSON-safe keys (spaces → underscores) used in the schema and model output
_to_key = {col: col.lower().replace(" ", "_") for col in LABEL_COLS}
_from_key = {v: k for k, v in _to_key.items()}
LABEL_KEYS: list[str] = [_to_key[c] for c in LABEL_COLS]

LABEL_DESCRIPTIONS: dict[str, str] = {
    "medical_material": (
        "Medical devices or foreign material visible in the chest (e.g., pacemaker, central line, "
        "port, stent, PICC, endotracheal tube, surgical clips/staples)."
    ),
    "arterial_wall_calcification": (
        "Calcification of the aortic wall, aortic arch, or major thoracic arterial walls."
    ),
    "cardiomegaly": "Enlarged cardiac silhouette / cardiomegaly.",
    "pericardial_effusion": "Pericardial fluid or effusion.",
    "coronary_artery_wall_calcification": "Calcification of the coronary arteries.",
    "hiatal_hernia": "Hiatal / hiatus hernia (stomach or bowel herniation through the diaphragm).",
    "lymphadenopathy": (
        "Enlarged or pathological lymph nodes in the mediastinum, hilum, or axilla."
    ),
    "emphysema": (
        "Pulmonary emphysema or air trapping (e.g., centrilobular, panlobular, paraseptal emphysema)."
    ),
    "atelectasis": (
        "Atelectasis, collapse, or subsegmental / linear / plate-like atelectasis in any lobe."
    ),
    "lung_nodule": (
        "Pulmonary nodule(s) — solid, part-solid, or ground-glass — of any size."
    ),
    "lung_opacity": (
        "Ground-glass opacity, hazy opacity, or non-specific pulmonary opacity in the lung parenchyma."
    ),
    "pulmonary_fibrotic_sequela": (
        "Fibrotic changes: honeycombing, traction bronchiectasis, subpleural fibrosis, "
        "reticular pattern consistent with prior or active fibrosis."
    ),
    "pleural_effusion": "Pleural fluid or effusion (unilateral or bilateral).",
    "mosaic_attenuation_pattern": (
        "Mosaic attenuation / perfusion pattern in the lung, suggesting air trapping or "
        "small-airway disease."
    ),
    "peribronchial_thickening": (
        "Peribronchial / bronchial wall thickening, peribronchial cuffing, or peribronchial infiltrates."
    ),
    "consolidation": (
        "Alveolar consolidation (airspace consolidation), including infectious, inflammatory, "
        "or malignant consolidation."
    ),
    "bronchiectasis": (
        "Bronchiectasis of any type (cylindrical, varicose, cystic) in any lobe."
    ),
    "interlobular_septal_thickening": (
        "Thickened interlobular septa, septal lines (Kerley B lines), or reticular pattern "
        "consistent with interstitial thickening."
    ),
}


# ── Model / client helpers ───────────────────────────────────────────────────

_DEEPSEEK_IDS = {"deepseekv3", "deepseekv32", "deepseekr1"}


def _is_deepseek(model_name: str) -> bool:
    normalised = re.sub(r"[-_. ]", "", model_name).lower()
    return normalised in _DEEPSEEK_IDS


def init_client(model_name: str, endpoint: str | None, api_key: str | None):
    endpoint = endpoint or os.getenv("AZURE_AI_ENDPOINT")
    api_key = api_key or os.getenv("AZURE_AI_KEY")
    if not endpoint or not api_key:
        raise RuntimeError(
            "Credentials missing. Pass --endpoint / --api-key or set "
            "AZURE_AI_ENDPOINT / AZURE_AI_KEY."
        )
    if _is_deepseek(model_name):
        client = OpenAI(base_url=endpoint, api_key=api_key)
    else:
        client = AzureOpenAI(
            api_version="2025-04-01-preview",
            azure_endpoint=endpoint,
            api_key=api_key,
        )
    return client, model_name


# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a radiology report classification system for an academic research study. "
    "Your sole task is to determine whether each of 18 predefined chest CT findings is "
    "mentioned as present in the report. "
    "Apply only what is explicitly stated; do not infer from clinical context or epidemiology. "
    "Return valid JSON matching the required schema exactly."
)

_LABEL_LIST = "\n".join(
    f'  {key}: {desc}'
    for key, desc in LABEL_DESCRIPTIONS.items()
)

USER_PROMPT_TEMPLATE = """\
Classify the following chest CT radiology report.

[FINDINGS]
{findings}

[IMPRESSION]
{impression}

CLASSIFICATION RULES
- Assign 1 (present) if the finding is clearly described as present, observed, or detected anywhere in [FINDINGS] or [IMPRESSION].
- Assign 0 (absent/not mentioned) if the finding is explicitly negated ("no …", "not detected", "absent") OR simply not mentioned.
- Base your decision solely on what is written. Do not infer.
- Uncertainty language ("cannot exclude", "possible", "suspected") should be assigned 1.

LABEL DEFINITIONS
{label_list}

Return a single JSON object with exactly these keys and integer values (0 or 1):
{{
{schema_example}
}}
"""


def build_user_prompt(findings: str, impression: str) -> str:
    schema_example = ",\n".join(f'  "{k}": 0' for k in LABEL_KEYS)
    return USER_PROMPT_TEMPLATE.format(
        findings=findings or "Not given.",
        impression=impression or "Not given.",
        label_list=_LABEL_LIST,
        schema_example=schema_example,
    )


# ── JSON schema for structured outputs (non-DeepSeek) ────────────────────────

def build_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {key: {"type": "integer", "enum": [0, 1]} for key in LABEL_KEYS},
        "required": LABEL_KEYS,
        "additionalProperties": False,
    }


RESPONSE_JSON_SCHEMA = build_json_schema()


# ── Output extraction helpers ─────────────────────────────────────────────────

def _extract_output_text(response) -> str | None:
    """Extract text from Responses API response (handles nested structure)."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    output = getattr(response, "output", None)
    if not output:
        return None
    chunks: list[str] = []
    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        for c in getattr(item, "content", None) or []:
            if getattr(c, "type", None) == "output_text":
                text = getattr(c, "text", None)
                if text:
                    chunks.append(text)
    return "".join(chunks) if chunks else None


def _safe_loads(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        s = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    return json.loads(s)


def _parse_labels(raw: str) -> dict[str, int] | None:
    try:
        obj = _safe_loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    result: dict[str, int] = {}
    for key in LABEL_KEYS:
        val = obj.get(key, 0)
        result[key] = 1 if (val == 1 or val is True) else 0
    return result


def _default_labels() -> dict[str, int]:
    return {key: 0 for key in LABEL_KEYS}


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_llm(
    client,
    model_name: str,
    findings: str,
    impression: str,
) -> tuple[dict[str, int] | None, dict]:
    user_content = build_user_prompt(findings, impression)

    if _is_deepseek(model_name):
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
    else:
        response = client.responses.create(
            model=model_name,
            reasoning={"effort": "low"},
            input=[
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_content}],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ct_rate_labels",
                    "strict": True,
                    "schema": RESPONSE_JSON_SCHEMA,
                }
            },
        )
        raw = _extract_output_text(response)

    if raw is None:
        return None, {"_raw_response": None, "_parse_error": "empty response"}

    labels = _parse_labels(raw)
    if labels is None:
        return None, {"_raw_response": raw, "_parse_error": "json parse failed"}

    return labels, {"_raw_response": raw, "_parse_error": ""}


# ── Main ──────────────────────────────────────────────────────────────────────

def _clean(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _model_slug(model_name: str) -> str:
    """Convert model name to a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", model_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign CT-RATE labels via LLM (Azure AI Foundry / DeepSeek)."
    )
    parser.add_argument(
        "--input-csv", default="data/CT-RATE_reports.csv",
        help="Path to merged CT-RATE CSV (default: data/CT-RATE_reports.csv).",
    )
    parser.add_argument(
        "--output-csv", default=None,
        help="Output path. Defaults to data/CT-RATE_reports_{model_name}.csv.",
    )
    parser.add_argument("--endpoint", default=None, help="Azure AI Foundry endpoint URL.")
    parser.add_argument("--api-key",  default=None, help="Azure AI Foundry API key.")
    parser.add_argument("--model-name", default="o3", help="Deployed model name.")
    parser.add_argument(
        "--findings-col",   default="Findings_EN",   help="Column for findings text.",
    )
    parser.add_argument(
        "--impression-col", default="Impressions_EN", help="Column for impression text.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N rows (useful for testing).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of parallel threads (default: 4).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="If output CSV already exists, skip rows that already have labels.",
    )
    args = parser.parse_args()

    slug = _model_slug(args.model_name)
    out_path = Path(args.output_csv) if args.output_csv else Path(f"data/CT-RATE_reports_{slug}.csv")

    client, model_name = init_client(args.model_name, args.endpoint, args.api_key)

    df = pd.read_csv(args.input_csv)
    for col in (args.findings_col, args.impression_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    # Resume: identify rows already processed in existing output
    already_done: set[int] = set()
    if args.resume and out_path.exists():
        existing = pd.read_csv(out_path)
        if "_extraction_failed" in existing.columns:
            done_mask = ~existing["_extraction_failed"].fillna(True).astype(bool)
            already_done = set(existing.index[done_mask].tolist())
            print(f"Resuming: {len(already_done):,} rows already processed, will skip them.")
            # Carry over existing labels into df
            for col in LABEL_COLS + ["_raw_response", "_parse_error", "_extraction_failed"]:
                if col in existing.columns:
                    df[col] = existing[col]

    if args.limit is not None:
        df = df.head(args.limit).copy()

    # Initialise output columns if not present
    for col in LABEL_COLS:
        if col not in df.columns:
            df[col] = 0
    for col in ("_raw_response", "_parse_error", "_extraction_failed"):
        if col not in df.columns:
            df[col] = None

    rows_to_process = [
        (i, row) for i, row in enumerate(df.to_dict(orient="records"))
        if i not in already_done
    ]
    print(
        f"Input  : {args.input_csv}  ({len(df):,} rows)\n"
        f"Output : {out_path}\n"
        f"Model  : {model_name}\n"
        f"Workers: {args.num_workers}\n"
        f"To process: {len(rows_to_process):,} rows"
    )

    def process_one(idx_row: tuple[int, dict]) -> tuple[int, dict[str, int], dict]:
        idx, row = idx_row
        findings  = _clean(row.get(args.findings_col))
        impression = _clean(row.get(args.impression_col))

        diag = {"_raw_response": "", "_parse_error": "", "_extraction_failed": False}

        if not findings and not impression:
            diag["_parse_error"] = "empty input"
            diag["_extraction_failed"] = True
            return idx, _default_labels(), diag

        try:
            labels, diag2 = call_llm(
                client=client,
                model_name=model_name,
                findings=findings,
                impression=impression,
            )
        except Exception as e:
            diag["_parse_error"] = f"api_error: {type(e).__name__}: {e}"
            diag["_extraction_failed"] = True
            return idx, _default_labels(), diag

        diag2["_extraction_failed"] = labels is None
        if labels is None:
            return idx, _default_labels(), diag2
        return idx, labels, diag2

    results: dict[int, dict[str, int]] = {}
    diags:   dict[int, dict]           = {}

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(process_one, ir) for ir in rows_to_process]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Labelling"):
            idx, labels, diag = fut.result()
            results[idx] = labels
            diags[idx]   = diag

    # Write results back into df
    for idx, labels in results.items():
        diag = diags[idx]
        for key, col in _from_key.items():
            df.at[idx, col] = labels[key]
        df.at[idx, "_raw_response"]     = diag.get("_raw_response", "")
        df.at[idx, "_parse_error"]      = diag.get("_parse_error", "")
        df.at[idx, "_extraction_failed"] = diag.get("_extraction_failed", False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    n_failed = int(df["_extraction_failed"].fillna(True).sum())
    print(
        f"\nDone.\n"
        f"  Saved       : {out_path}\n"
        f"  Total rows  : {len(df):,}\n"
        f"  Failed      : {n_failed:,} ({n_failed / len(df) * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
