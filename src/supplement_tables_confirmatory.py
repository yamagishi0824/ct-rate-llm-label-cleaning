"""確認セットの補遺用カテゴリ別表（Supplementary Table 4）。API は使わない。

開発セットの tables/model_vs_human_per_label.csv と同じ列構成で、Rule A の確認セットについて
TP/FP/FN/TN、有病率、各指標と 95% CI を出す。比率の CI は Wilson 法、F1 と κ の CI は
report 単位 bootstrap（層内復元抽出、2,000 回）。design-weighted の F1 と κ も併記する。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_confirmatory import LABELS, counts, load, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def wilson(k, n, z=norm.ppf(0.975)):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def main() -> int:
    vols, R, src, w, strata = load("A")
    ones = np.ones(len(vols))
    rng = np.random.default_rng(20261020)
    idx_by = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    boots = [np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by]) for _ in range(2000)]
    rows = []
    for name, S in src.items():
        tp, fp, fn, tn = counts(R, S, ones); m = metrics(tp, fp, fn, tn)
        mw = metrics(*counts(R, S, w))
        bf = np.empty((len(boots), len(LABELS))); bk = np.empty_like(bf)
        for b, ix in enumerate(boots):
            mm = metrics(*counts(R[ix], S[ix], ones[ix])); bf[b] = mm["f1"]; bk[b] = mm["kappa"]
        for j, lab in enumerate(LABELS):
            T, F, N_, Tn = int(tp[j]), int(fp[j]), int(fn[j]), int(tn[j]); n = T + F + N_ + Tn
            r = {"Source": name, "Label": lab, "N": n, "TP": T, "FP": F, "FN": N_, "TN": Tn,
                 "Prev_reference_%": round((T + N_) / n * 100, 1), "Prev_source_%": round((T + F) / n * 100, 1)}
            for key, k_, d_ in [("Sensitivity", T, T + N_), ("Specificity", Tn, Tn + F), ("PPV", T, T + F),
                                ("NPV", Tn, Tn + N_), ("Accuracy", T + Tn, n)]:
                lo, hi = wilson(k_, d_); r[key] = k_ / d_ if d_ else np.nan; r[f"{key}_lo"] = lo; r[f"{key}_hi"] = hi
            r["F1"] = m["f1"][j]; r["F1_lo"], r["F1_hi"] = np.nanpercentile(bf[:, j], [2.5, 97.5])
            r["Kappa"] = m["kappa"][j]; r["Kappa_lo"], r["Kappa_hi"] = np.nanpercentile(bk[:, j], [2.5, 97.5])
            r["F1_weighted"] = mw["f1"][j]; r["Kappa_weighted"] = mw["kappa"][j]
            rows.append(r)
    out = ROOT / "tables/confirmatory_per_label_full_ruleA.csv"
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print(f"{len(rows)} 行 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
