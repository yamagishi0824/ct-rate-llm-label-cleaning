"""4値版の抽出（present / uncertain / negated / not_mentioned）。

ラベル定義（LABEL_DESCRIPTIONS）とモデル設定は本番の二値抽出 src/extract_labels.py と
共通で、変えるのは出力の値域と、その値の判定規則だけ。値の定義は放射線科医の
4値アノテーションと同じ。
"""
from __future__ import annotations
from extract_labels import (LABEL_DESCRIPTIONS, LABEL_KEYS, _extract_output_text,
                            _is_deepseek, _safe_loads)

VALUES = ["present", "uncertain", "negated", "not_mentioned"]

SYSTEM_PROMPT_4 = (
    "You are a radiology report classification system for an academic research study. "
    "Your sole task is to assign one of four values to each of 18 predefined chest CT findings "
    "based on the report. "
    "Apply only what is explicitly stated; do not infer from clinical context or epidemiology. "
    "Return valid JSON matching the required schema exactly."
)

_LABEL_LIST = "\n".join(f"  {k}: {d}" for k, d in LABEL_DESCRIPTIONS.items())

USER_PROMPT_TEMPLATE_4 = """\
Classify the following chest CT radiology report.

[FINDINGS]
{findings}

[IMPRESSION]
{impression}

CLASSIFICATION RULES
Assign exactly one of these values to every finding, based on [FINDINGS] and [IMPRESSION]:
- "present": the finding is described as present, observed, or detected.
- "uncertain": the finding is mentioned with hedged language ("possible", "suspected", "cannot exclude",
  "may represent", "?", "in favor of" with explicit doubt).
- "negated": the finding is explicitly stated to be absent ("no ...", "not detected", "absent").
- "not_mentioned": the finding is not referred to at all.
Base your decision solely on what is written. Do not infer.

LABEL DEFINITIONS
{label_list}

Return a single JSON object with exactly these keys, each set to one of
"present", "uncertain", "negated", "not_mentioned":
{{
{schema_example}
}}
"""


def build_user_prompt_4(findings: str, impression: str) -> str:
    ex = ",\n".join(f'  "{k}": "not_mentioned"' for k in LABEL_KEYS)
    return USER_PROMPT_TEMPLATE_4.format(findings=findings or "Not given.",
                                         impression=impression or "Not given.",
                                         label_list=_LABEL_LIST, schema_example=ex)


SCHEMA_4 = {
    "type": "object",
    "properties": {k: {"type": "string", "enum": VALUES} for k in LABEL_KEYS},
    "required": list(LABEL_KEYS),
    "additionalProperties": False,
}


def parse_4(raw: str) -> dict[str, str] | None:
    try:
        obj = _safe_loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    out = {}
    for k in LABEL_KEYS:
        v = str(obj.get(k, "")).strip().lower().replace(" ", "_")
        if v not in VALUES:
            return None
        out[k] = v
    return out


def call_llm_4(client, model_name: str, findings: str, impression: str):
    user = build_user_prompt_4(findings, impression)
    if _is_deepseek(model_name):
        r = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": SYSTEM_PROMPT_4},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        raw = r.choices[0].message.content
    else:
        r = client.responses.create(
            model=model_name,
            reasoning={"effort": "low"},
            input=[{"role": "developer", "content": [{"type": "input_text", "text": SYSTEM_PROMPT_4}]},
                   {"role": "user", "content": [{"type": "input_text", "text": user}]}],
            text={"format": {"type": "json_schema", "name": "ct_rate_labels_4level",
                             "strict": True, "schema": SCHEMA_4}},
        )
        raw = _extract_output_text(r)
    if raw is None:
        return None, {"_raw_response": None, "_parse_error": "empty response"}
    labels = parse_4(raw)
    if labels is None:
        return None, {"_raw_response": raw, "_parse_error": "json parse failed or invalid value"}
    return labels, {"_raw_response": raw, "_parse_error": ""}
