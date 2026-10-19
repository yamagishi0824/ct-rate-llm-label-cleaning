"""二値化ルールの感度分析（PH-05）。API は使わない。

Rule A（主）: present + uncertain を陽性。Rule B: present のみ陽性。
Rule C: uncertain のインスタンスを除外して Rule A で評価。
Rule D: 参照標準の陰性を negated と not_mentioned に分け、それぞれでの偽陽性率を出す
        （「言及なし＝陰性」という CT-RATE の規約の検証）。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_confirmatory import LABELS, load, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    vols, RA, src, w, strata = load("A")
    _, RB, _, _, _ = load("B")
    ref4 = (pd.read_csv(ROOT / "data/confirmatory250_reference_standard.csv")
            .pivot(index="volume_name", columns="label_name", values="consensus").loc[vols, LABELS].to_numpy())
    ones = np.ones(len(vols)); rows = []
    keep = ref4 != "uncertain"
    for name, S in src.items():
        _, a, _, _ = summarize(RA, S, ones); _, b, _, _ = summarize(RB, S, ones)
        # Rule C: uncertain を除外。ラベル別に除外して macro を取る
        f1c, kc = [], []
        for j in range(len(LABELS)):
            k = keep[:, j]
            _, m, _, _ = summarize(RA[k, j:j+1], S[k, j:j+1], ones[k]); f1c.append(m["f1"]); kc.append(m["kappa"])
        neg, nm = ref4 == "negated", ref4 == "not_mentioned"
        rows.append({"source": name,
                     "A_f1": a["f1"], "A_kappa": a["kappa"], "B_f1": b["f1"], "B_kappa": b["kappa"],
                     "C_f1": np.nanmean(f1c), "C_kappa": np.nanmean(kc),
                     "D_fpr_among_negated": (S[neg] == 1).mean(), "D_fpr_among_not_mentioned": (S[nm] == 1).mean(),
                     "D_n_negated": int(neg.sum()), "D_n_not_mentioned": int(nm.sum())})
    T = pd.DataFrame(rows); T.to_csv(ROOT / "tables/confirmatory_sensitivity_rules.csv", index=False)
    for _, r in T.iterrows():
        print(f"{r.source:14s} A F1 {r.A_f1*100:5.1f} k {r.A_kappa:.3f} | B F1 {r.B_f1*100:5.1f} k {r.B_kappa:.3f} | "
              f"C F1 {r.C_f1*100:5.1f} k {r.C_kappa:.3f} | D 偽陽性率 negated {r.D_fpr_among_negated*100:4.2f}% / not_mentioned {r.D_fpr_among_not_mentioned*100:4.2f}%")
    print(f"\n(D の分母: negated {T.D_n_negated.iloc[0]} / not_mentioned {T.D_n_not_mentioned.iloc[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
