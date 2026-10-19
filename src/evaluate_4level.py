"""4値出力の LLM 評価。API は使わない。

data/repeat_runs/<model>_4level_run1.csv を参照標準（合議値、4値）と比べる。
  - 4値の完全一致率と Cohen κ（全体・カテゴリ別）
  - 混同行列（参照標準 × モデル、全インスタンス合算）
  - Rule A / Rule B で二値化したときの label-macro F1 / κ（二値形式のラン1との比較つき）
  - 参照標準が陰性（negated / not_mentioned）のインスタンスで、両者を区別できているか
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_confirmatory import LABELS, RULES, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"gpt-5.4": "GPT-5.4", "gpt-5.4-mini": "GPT-5.4 mini", "DeepSeek-V3.2": "DeepSeek V3.2"}
VALS = ["present", "uncertain", "negated", "not_mentioned"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=ROOT / "data/repeat_runs")
    ap.add_argument("--out", type=Path, default=ROOT / "tables")
    args = ap.parse_args()

    vols = pd.read_csv(ROOT / "data/confirmatory250_design.csv").sort_values("reading_order").VolumeName.tolist()
    ref = (pd.read_csv(ROOT / "data/confirmatory250_reference_standard.csv")
           .pivot(index="volume_name", columns="label_name", values="consensus").loc[vols, LABELS])
    ones = np.ones(len(vols))

    summ, per, conf = [], [], []
    for slug, name in MODELS.items():
        f = args.run_dir / f"{slug}_4level_run1.csv"
        if not f.exists():
            print(f"  未生成: {f.name}"); continue
        d = pd.read_csv(f).set_index("VolumeName").reindex(vols)
        nfail = int(d["_extraction_failed"].astype(bool).sum())
        M = d[LABELS]
        r, m = ref.to_numpy().ravel(), M.to_numpy().ravel()
        ok = pd.notna(m)
        row = {"model": name, "failed_reports": nfail,
               "exact_4level": (r[ok] == m[ok]).mean(),
               "kappa_4level": cohen_kappa_score(r[ok], m[ok], labels=VALS)}
        for rule in ("A", "B"):
            R = ref.replace(RULES[rule]).to_numpy(float)
            S = M.replace(RULES[rule]).fillna(0).to_numpy(float)
            _, macro, _, _ = summarize(R, S, ones)
            row[f"macro_f1_rule{rule}"], row[f"macro_kappa_rule{rule}"] = macro["f1"], macro["kappa"]
        b = args.run_dir / f"{slug}_run1.csv"
        if b.exists():
            B = pd.read_csv(b).set_index("VolumeName").reindex(vols)[LABELS].to_numpy(float)
            _, macro, _, _ = summarize(ref.replace(RULES["A"]).to_numpy(float), B, ones)
            row["binary_scheme_macro_f1"], row["binary_scheme_macro_kappa"] = macro["f1"], macro["kappa"]
        neg = np.isin(r, ["negated", "not_mentioned"]) & np.isin(m, ["negated", "not_mentioned"])
        row["neg_vs_nm_agreement"] = (r[neg] == m[neg]).mean()
        row["neg_vs_nm_kappa"] = cohen_kappa_score(r[neg], m[neg])
        summ.append(row)
        for lab in LABELS:
            a, bb = ref[lab].to_numpy(), M[lab].to_numpy()
            k = pd.notna(bb)
            per.append({"model": name, "label": lab, "exact_4level": (a[k] == bb[k]).mean(),
                        "kappa_4level": cohen_kappa_score(a[k], bb[k], labels=VALS)})
        ct = pd.crosstab(pd.Series(r[ok], name="reference"), pd.Series(m[ok], name="model")).reindex(index=VALS, columns=VALS, fill_value=0)
        ct.insert(0, "model_name", name)
        conf.append(ct.reset_index())

    if not summ:
        return 1
    args.out.mkdir(exist_ok=True)
    S = pd.DataFrame(summ); S.to_csv(args.out / "llm_4level_summary.csv", index=False)
    pd.DataFrame(per).to_csv(args.out / "llm_4level_per_label.csv", index=False)
    pd.concat(conf).to_csv(args.out / "llm_4level_confusion.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.round(4).to_string(index=False))
    for c in conf:
        print(f"\n■ 混同行列 {c.model_name.iloc[0]}（行=参照標準, 列=モデル）")
        print(c.drop(columns="model_name").set_index("reference").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
