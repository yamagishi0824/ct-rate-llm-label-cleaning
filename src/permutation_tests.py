"""全体の対応比較を、レポート内相関を保ったまま検定する（レポート単位の並べ替え検定）。

各レポートについて2つのラベルソースの出力（18ラベルまとめて）を確率 1/2 で入れ替え、
label-macro F1 の差の帰無分布を作る。両側 p = |差| が観測値以上となる割合。
確認セットは1患者1レポートなので、患者単位の並べ替えと同等。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_confirmatory import LABELS, counts, load, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [("Majority vote", "CT-RATE"), ("GPT-5.4", "CT-RATE"), ("GPT-5.4 mini", "CT-RATE"),
         ("DeepSeek V3.2", "CT-RATE"), ("Majority vote", "GPT-5.4"), ("Majority vote", "GPT-5.4 mini"),
         ("Majority vote", "DeepSeek V3.2")]


def macro_f1(R, S):
    return np.nanmean(metrics(*counts(R, S, np.ones(len(R))))["f1"])


def perm(R, A, B, n=10000, seed=20261020):
    rng = np.random.default_rng(seed)
    obs = macro_f1(R, A) - macro_f1(R, B)
    ge = 0
    for _ in range(n):
        sw = rng.random(len(R)) < 0.5
        A2 = np.where(sw[:, None], B, A); B2 = np.where(sw[:, None], A, B)
        ge += abs(macro_f1(R, A2) - macro_f1(R, B2)) >= abs(obs) - 1e-12
    return obs, (ge + 1) / (n + 1)


def development():
    a = pd.read_csv(ROOT / "data/random_annotations_v2.csv"); a = a[a.annotator == "髙松 篤"]
    vols = sorted(a.volume_name.unique())
    R = a.pivot(index="volume_name", columns="label_name", values="value").loc[vols, LABELS].astype(float).to_numpy()
    src = {"CT-RATE": pd.read_csv(ROOT / "data/CT-RATE_reports.csv").set_index("VolumeName").loc[vols, LABELS].to_numpy(float)}
    for n, s in [("GPT-5.4", "gpt-5.4"), ("GPT-5.4 mini", "gpt-5.4-mini"), ("DeepSeek V3.2", "DeepSeek-V3.2")]:
        src[n] = pd.read_csv(ROOT / f"data/CT-RATE_reports_{s}.csv").set_index("VolumeName").loc[vols, LABELS].to_numpy(float)
    src["Majority vote"] = (np.stack([src[m] for m in ["GPT-5.4", "GPT-5.4 mini", "DeepSeek V3.2"]]).sum(0) >= 2).astype(float)
    return R, src


def main():
    rows = []
    _, Rc, srcc, _, _ = load("A")
    Rd, srcd = development()
    for setname, R, src in [("development", Rd, srcd), ("confirmatory", Rc, srcc)]:
        for a, b in PAIRS:
            d, p = perm(R, src[a], src[b])
            rows.append({"set": setname, "comparison": f"{a} vs {b}", "diff_macro_f1": d, "perm_p": p})
            print(f"{setname:12s} {a:14s} vs {b:14s} ΔF1 {d*100:+.1f}  permutation p = {p:.4f}")
    pd.DataFrame(rows).to_csv(ROOT / "tables/paired_permutation_tests.csv", index=False)


if __name__ == "__main__":
    main()
