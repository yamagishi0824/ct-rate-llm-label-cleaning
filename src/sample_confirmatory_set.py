#!/usr/bin/env python3
"""sample_confirmatory_set.py

JOMS revision 用 confirmatory reference set（250レポート）の抽出。

設計（Methods にそのまま書ける形）
--------------------------------
1. 母集団: CT-RATE validation split。report-level 重複除去後 1,470 レポート / 1,224 患者。
   - 既存アノテーション（random / general / focus, 計297レポート）に含まれる患者を除外
   - 1患者1レポート（VolumeName 昇順で最初のもの）
   → sampling frame 1,212 レポート = 1,212 患者。report-level = patient-level となるため
     クラスタ相関の問題が設計段階で消える。

2. 層化（2層のみ）: 低頻度カテゴリ（validation split での有病率 <15% の9カテゴリ）の
   いずれかを **既存の CT-RATE ラベル** が陽性としているか否かで2層に分割。
     stratum 1 = 低頻度カテゴリの CT-RATE 陽性を1つ以上含む
     stratum 2 = 含まない
   stratum 1 から 190、stratum 2 から 60 を層内単純無作為抽出。
   design weight = N_h / n_h。

   層化変数に本研究の LLM 出力を使わないのは、confirmatory test set の構成に
   評価対象モデルの出力が関与することを避けるため（CT-RATE ラベルは公開済みの既存情報）。
   4ソースの union で層化した場合と比較して、各カテゴリの陽性インスタンス数は
   ほぼ同等（最少カテゴリで 24 対 25）。

注意
----
層化には公開済みの CT-RATE ラベルのみを使用し、本研究の LLM 出力は使用しない。
読影者には
  各カテゴリの CT-RATE / LLM ラベル、層の割り当て、不一致の方向
を一切開示しない（blinded independent annotation）。読影順はシャッフル済み。

出力
----
data/confirmatory250_design.csv   … 内部用（層・重み・読影順）。読影者には渡さない。

Usage
-----
  python src/sample_confirmatory_set.py     # SEED 固定で再現可能
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260921
N_ALLOC = {1: 190, 2: 60}          # 合計 250
LOW_PREV_THRESHOLD = 0.15          # validation split での CT-RATE 有病率

DATA = Path(__file__).resolve().parent.parent / "data"
MODEL_FILES = {
    "CT-RATE": "CT-RATE_reports.csv",
    "GPT-5.4": "CT-RATE_reports_gpt-5.4.csv",
    "GPT-5.4 mini": "CT-RATE_reports_gpt-5.4-mini.csv",
    "DeepSeek-V3.2": "CT-RATE_reports_DeepSeek-V3.2.csv",
}
ANNOTATED = [
    "random_annotations_v2.csv",
    "general_annotations_v2.csv",
    "focus_annotations_v2.csv",
]
META = ["VolumeName", "ClinicalInformation_EN", "Technique_EN",
        "Findings_EN", "Impressions_EN", "split"]


def main() -> None:
    ref = pd.read_csv(DATA / MODEL_FILES["CT-RATE"])
    labels = [c for c in ref.columns if c not in META]

    # ── 1. sampling frame ────────────────────────────────────────
    valid = ref[ref["split"] == "valid"].copy()
    valid["patient_id"] = valid["VolumeName"].str.extract(r"^(valid_\d+)")

    already: set[str] = set()
    annotated_volumes: set[str] = set()
    for f in ANNOTATED:
        for v in pd.read_csv(DATA / f)["volume_name"].unique():
            annotated_volumes.add(v)
            already.add(re.match(r"^((?:train|valid)_\d+)", v).group(1))

    frame = (valid[~valid["patient_id"].isin(already)]
             .sort_values("VolumeName")
             .groupby("patient_id", as_index=False)
             .first())

    # ── 2. 層化変数（公開済み CT-RATE ラベルのみを使用） ────────────
    ct_labels = (pd.read_csv(DATA / MODEL_FILES["CT-RATE"], usecols=["VolumeName"] + labels)
                 .set_index("VolumeName")
                 .reindex(frame["VolumeName"])[labels]
                 .fillna(0).values.astype(int))

    prevalence = ref.loc[ref["split"] == "valid", labels].mean()
    low_prev = [l for l in labels if prevalence[l] < LOW_PREV_THRESHOLD]
    low_idx = [labels.index(l) for l in low_prev]

    has_low = ct_labels[:, low_idx].sum(axis=1) >= 1
    frame = frame.assign(stratum=np.where(has_low, 1, 2))

    # 参考値の算出にのみ、4ソースの union を使う（層化には使わない）
    union = np.zeros((len(frame), len(labels)), dtype=int)
    for fname in MODEL_FILES.values():
        src = (pd.read_csv(DATA / fname, usecols=["VolumeName"] + labels)
               .set_index("VolumeName")
               .reindex(frame["VolumeName"])[labels]
               .fillna(0).values.astype(int))
        union |= src

    # ── 3. 層内単純無作為抽出 ────────────────────────────────────
    rng = np.random.default_rng(SEED)
    picked = []
    for stratum, n in N_ALLOC.items():
        sub = frame[frame["stratum"] == stratum]
        if len(sub) < n:
            raise ValueError(f"stratum {stratum}: 母数 {len(sub)} < 抽出数 {n}")
        take = sub.iloc[rng.choice(len(sub), n, replace=False)].copy()
        take["stratum_N"] = len(sub)
        take["stratum_n"] = n
        take["design_weight"] = len(sub) / n
        picked.append(take)

    sample = pd.concat(picked, ignore_index=True)
    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    sample.insert(0, "reading_order", np.arange(1, len(sample) + 1))

    out = sample[["reading_order", "VolumeName", "patient_id", "stratum",
                  "stratum_N", "stratum_n", "design_weight"]]
    out.to_csv(DATA / "confirmatory250_design.csv", index=False)

    # ── ログ ─────────────────────────────────────────────────────
    print(f"SEED = {SEED}")
    print(f"\n低頻度カテゴリ（validation split 有病率 <{LOW_PREV_THRESHOLD:.0%}）: {len(low_prev)}")
    for l in sorted(low_prev, key=lambda x: prevalence[x]):
        print(f"    {l:<36s} {prevalence[l]*100:4.1f}%")

    print(f"\nsampling frame: {len(frame)} レポート = {frame['patient_id'].nunique()} 患者")
    print(f"  （valid split 1,470レポート → 既存アノテーション患者を除外 → 1患者1レポート）")

    w = sample["design_weight"].values
    ess = w.sum() ** 2 / (w ** 2).sum()
    print("\n抽出結果")
    print(out.groupby("stratum").agg(n=("VolumeName", "size"), N=("stratum_N", "first"),
                                     design_weight=("design_weight", "first")).to_string())
    print(f"  合計 {len(out)} レポート / {out['patient_id'].nunique()} 患者")
    print(f"  重み付き推定の有効サンプルサイズ (Kish ESS) = {ess:.0f}")
    print(f"  既存アノテーション済みレポートとの重複 = "
          f"{len(set(out['VolumeName']) & annotated_volumes)}")

    mask = frame["VolumeName"].isin(out["VolumeName"]).values
    counts = pd.Series(union[mask].sum(axis=0), index=labels)
    summary = pd.DataFrame({
        "valid_prevalence_%": (prevalence * 100).round(1),
        "union_positive_in_250": counts,
    }).sort_values("valid_prevalence_%")
    print("\n抽出250例における union 陽性インスタンス数（読影者判定前の参考値）")
    print(summary.to_string())
    print(f"\n→ {DATA / 'confirmatory250_design.csv'}")


if __name__ == "__main__":
    main()
