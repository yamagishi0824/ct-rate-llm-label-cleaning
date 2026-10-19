#!/usr/bin/env python3
"""qa_ja_translations.py

confirmatory set の日本語訳（data/confirmatory250_reports_ja.json）の機械的な品質検査。

検査項目
--------
1. measurement   英語原文に出てくる計測値（mm / cm）が訳文に現れているか（訳抜けの検出）
2. sentence      文数の比（原文の文が落ちていないか）
3. laterality    right / left の出現数と 右 / 左 の出現数の一致
4. pleural_pc    pleural と pericardial の語数が、訳文の 胸膜系 / 心囊系 の語数と整合するか
                 （"No pleural effusion" を「心囊液は…」と訳す取り違えの検出）
5. length        文字数比が分布から外れていないか

出力
----
- 標準出力に集計
- data/llm4_pilot/ja_qa_flags.csv   レポート別のフラグ
- /tmp/ja_retranslate.txt           再翻訳対象の VolumeName 一覧

Usage
-----
  python src/qa_ja_translations.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MEAS = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mm|cm)\b", re.I)
NUM = re.compile(r"\d+(?:[.,]\d+)?")


def norm_num(x: str) -> str:
    return x.replace(",", ".").rstrip("0").rstrip(".") if "." in x.replace(",", ".") else x


def main() -> None:
    ja = json.loads((DATA / "confirmatory250_reports_ja.json").read_text(encoding="utf-8"))
    rep = (pd.read_csv(DATA / "CT-RATE_reports.csv",
                       usecols=["VolumeName", "Findings_EN", "Impressions_EN"])
           .set_index("VolumeName"))

    rows = []
    for v, t in ja.items():
        en = f"{rep.loc[v, 'Findings_EN']} {rep.loc[v, 'Impressions_EN']}"
        jp = f"{t['findings']} {t['impression']}"

        # 1. 計測値
        en_meas = {norm_num(m.group(1)) for m in MEAS.finditer(en)}
        jp_nums = {norm_num(m.group(0)) for m in NUM.finditer(jp)}
        missing = sorted(en_meas - jp_nums)

        # 2. 文数
        n_en = len([s for s in re.split(r"(?<=[.;])\s+", en) if len(s.strip()) > 3])
        n_jp = len([s for s in re.split(r"[。；]", jp) if len(s.strip()) > 1])

        # 3. 左右
        r_en = len(re.findall(r"\bright\b", en, re.I))
        l_en = len(re.findall(r"\bleft\b", en, re.I))
        r_jp = jp.count("右")
        l_jp = jp.count("左")

        # 4. pleural / pericardial の取り違え
        pl_en = len(re.findall(r"pleur", en, re.I))
        pc_en = len(re.findall(r"pericardi", en, re.I))
        pl_jp = len(re.findall(r"胸膜|胸水|胸腔", jp))
        pc_jp = len(re.findall(r"心囊|心嚢|心膜", jp))

        rows.append(dict(
            volume_name=v,
            missing_meas=len(missing), missing_list=";".join(missing),
            n_en=n_en, n_jp=n_jp, sent_ratio=round(n_jp / n_en, 2) if n_en else None,
            right_en=r_en, right_jp=r_jp, left_en=l_en, left_jp=l_jp,
            pleural_en=pl_en, pleural_jp=pl_jp, pericard_en=pc_en, pericard_jp=pc_jp,
            len_en=len(en), len_jp=len(jp), len_ratio=round(len(jp) / len(en), 3),
        ))

    d = pd.DataFrame(rows)

    # フラグ判定
    d["flag_meas"] = d.missing_meas > 0
    d["flag_sent"] = (d.sent_ratio < 0.8) | (d.sent_ratio > 1.3)
    d["flag_lat"] = ((d.right_en > 0) & (d.right_jp == 0)) | ((d.left_en > 0) & (d.left_jp == 0))
    d["flag_pl"] = ((d.pleural_en > 0) & (d.pleural_jp == 0)) | \
                   ((d.pericard_en > 0) & (d.pericard_jp == 0)) | \
                   ((d.pleural_en >= 2) & (d.pleural_jp <= d.pleural_en - 2))
    lo, hi = d.len_ratio.quantile([0.02, 0.98])
    d["flag_len"] = (d.len_ratio < lo) | (d.len_ratio > hi)
    flags = [c for c in d.columns if c.startswith("flag_")]
    d["n_flags"] = d[flags].sum(axis=1)

    d.to_csv(DATA / "llm4_pilot" / "ja_qa_flags.csv", index=False)

    print("=== 日本語訳 QA（250レポート）===")
    print(f"  文字数比 中央値 {d.len_ratio.median():.2f}（2〜98パーセンタイル {lo:.2f}–{hi:.2f}）")
    print(f"  文数比   中央値 {d.sent_ratio.median():.2f}")
    print()
    for c in flags:
        print(f"  {c:12s} {int(d[c].sum()):3d} 件")
    print(f"\n  フラグ1個以上: {int((d.n_flags > 0).sum())} 件 / 250")
    print(f"  フラグ2個以上: {int((d.n_flags > 1).sum())} 件")

    print("\n--- 計測値の訳抜け（上位） ---")
    for _, r in d[d.flag_meas].sort_values("missing_meas", ascending=False).head(8).iterrows():
        print(f"  {r.volume_name}  欠落 {r.missing_meas} 個: {r.missing_list[:60]}")

    print("\n--- pleural / pericardial の不整合 ---")
    for _, r in d[d.flag_pl].head(10).iterrows():
        print(f"  {r.volume_name}  EN pleural={r.pleural_en} pericardial={r.pericard_en} / "
              f"JA 胸膜系={r.pleural_jp} 心囊系={r.pericard_jp}")

    todo = d[d.n_flags > 0].volume_name.tolist()
    Path("/tmp/ja_retranslate.txt").write_text("\n".join(todo))
    print(f"\n再翻訳・確認の対象 {len(todo)} 件 → /tmp/ja_retranslate.txt")


if __name__ == "__main__":
    main()
