"""confirmatory250 の対応比較（PH-03 / PH-04）。API は使わない。

PH-03: 参照標準のインスタンス数と4値の分布。
PH-04: 各ラベルソース vs CT-RATE（および多数決 vs 各モデル）について
  - 誤りの対応 2×2 に対する exact McNemar 検定（全インスタンス合算、カテゴリ別 + Holm 補正）
  - label-macro F1 と Cohen κ の対応差（report 単位 bootstrap 2,000 回、層内復元抽出）
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_confirmatory import LABELS, load, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [("Majority vote", "CT-RATE"), ("GPT-5.4", "CT-RATE"), ("GPT-5.4 mini", "CT-RATE"),
         ("DeepSeek V3.2", "CT-RATE"), ("Majority vote", "GPT-5.4"),
         ("Majority vote", "GPT-5.4 mini"), ("Majority vote", "DeepSeek V3.2")]


def mcnemar(err_a, err_b):
    """a が正解で b が誤り（b_wrong）と、その逆（a_wrong）の exact 検定。"""
    a_wrong = int((err_a & ~err_b).sum()); b_wrong = int((~err_a & err_b).sum())
    n = a_wrong + b_wrong
    p = binomtest(a_wrong, n, 0.5).pvalue if n else 1.0
    return a_wrong, b_wrong, p


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(run, 1.0)
    return adj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--out", type=Path, default=ROOT / "tables")
    args = ap.parse_args()

    # PH-03
    ref = pd.read_csv(ROOT / "data/confirmatory250_reference_standard.csv")
    dist = ref.consensus.value_counts().reindex(["present", "uncertain", "negated", "not_mentioned"]).fillna(0).astype(int)
    print(f"■ PH-03 参照標準: {ref.volume_name.nunique()} reports × {ref.label_name.nunique()} labels = {len(ref)} instances")
    for k, v in dist.items(): print(f"    {k:14s} {v:5d} ({v/len(ref)*100:5.2f}%)")
    unc = ref[ref.consensus == "uncertain"].label_name.value_counts()
    print(f"    uncertain のカテゴリ内訳: {unc.to_dict()}\n")

    vols, R, src, w, strata = load("A")
    err = {k: (S != R) for k, S in src.items()}
    rows, cat_rows = [], []
    rng = np.random.default_rng(args.seed)
    idx_by = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    boots = [np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by]) for _ in range(args.boot)]
    ones = np.ones(len(vols))
    for a, b in PAIRS:
        aw, bw, p = mcnemar(err[a], err[b])
        _, ma, _, _ = summarize(R, src[a], ones); _, mb, _, _ = summarize(R, src[b], ones)
        dF, dK = ma["f1"] - mb["f1"], ma["kappa"] - mb["kappa"]
        bf, bk = [], []
        for ix in boots:
            _, x, _, _ = summarize(R[ix], src[a][ix], ones[ix]); _, y, _, _ = summarize(R[ix], src[b][ix], ones[ix])
            bf.append(x["f1"] - y["f1"]); bk.append(x["kappa"] - y["kappa"])
        rows.append({"comparison": f"{a} vs {b}", f"errors_{a}": aw + 0, "errors_only_a": aw, "errors_only_b": bw,
                     "mcnemar_p": p, "diff_macro_f1": dF, "diff_f1_lo": np.percentile(bf, 2.5), "diff_f1_hi": np.percentile(bf, 97.5),
                     "diff_macro_kappa": dK, "diff_k_lo": np.percentile(bk, 2.5), "diff_k_hi": np.percentile(bk, 97.5)})
        ps = []
        for j, lab in enumerate(LABELS):
            x, y, pj = mcnemar(err[a][:, j], err[b][:, j]); ps.append(pj)
            cat_rows.append({"comparison": f"{a} vs {b}", "label": lab, "errors_only_a": x, "errors_only_b": y, "p": pj})
        adj = holm(ps)
        for j in range(len(LABELS)): cat_rows[-len(LABELS) + j]["p_holm"] = adj[j]

    T = pd.DataFrame(rows).drop(columns=[c for c in pd.DataFrame(rows).columns if c.startswith("errors_") and c not in ("errors_only_a", "errors_only_b")])
    C = pd.DataFrame(cat_rows)
    args.out.mkdir(exist_ok=True)
    T.to_csv(args.out / "confirmatory_paired_ruleA.csv", index=False)
    C.to_csv(args.out / "confirmatory_paired_per_label_ruleA.csv", index=False)
    print("■ PH-04 対応比較（Rule A、非重み付け）  errors_only_a = a だけが誤り")
    for _, r in T.iterrows():
        print(f"  {r.comparison:30s} 誤り {int(r.errors_only_a):3d} vs {int(r.errors_only_b):3d}  p={r.mcnemar_p:.2g}"
              f"  ΔF1 {r.diff_macro_f1*100:+.1f} ({r.diff_f1_lo*100:+.1f}, {r.diff_f1_hi*100:+.1f})"
              f"  Δκ {r.diff_macro_kappa:+.3f} ({r.diff_k_lo:+.3f}, {r.diff_k_hi:+.3f})")
    sig = C[(C.p_holm < 0.05)]
    print(f"\n  カテゴリ別で Holm 補正後 p<0.05: {len(sig)} 件")
    for _, r in sig.iterrows(): print(f"    {r.comparison:30s} {r.label:34s} {int(r.errors_only_a)} vs {int(r.errors_only_b)}  p_holm={r.p_holm:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
