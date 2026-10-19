"""反復ランの集計（PH-10）。API は使わない。

data/repeat_runs/<model>_run<k>.csv を読み、モデルごとに
  - instance-level flip rate（ラン間で値が一致しないインスタンスの割合）全体・カテゴリ別、
    非重み付けとデザイン重み付け
  - ラン間の Fleiss κ（全体・カテゴリ別）
  - 各ランの参照標準に対する label-macro F1 / κ（Rule A）
  - 所要時間（logs/repeat_runs_summary.csv）
を出す。3 モデルが揃ったランについては多数決のラン間 flip rate も出す。
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_confirmatory import LABELS, RULES, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"gpt-5.4": "GPT-5.4", "gpt-5.4-mini": "GPT-5.4 mini", "DeepSeek-V3.2": "DeepSeek V3.2"}


def fleiss_kappa(X):
    """X: (instances, runs) の 0/1。2 カテゴリの Fleiss κ。"""
    n = X.shape[1]
    p1 = X.mean(1)
    P_i = (p1 * n * (p1 * n - 1) + (1 - p1) * n * ((1 - p1) * n - 1)) / (n * (n - 1))
    P_bar = P_i.mean()
    pj = X.mean()
    P_e = pj ** 2 + (1 - pj) ** 2
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=ROOT / "data/repeat_runs")
    ap.add_argument("--out", type=Path, default=ROOT / "tables")
    args = ap.parse_args()

    design = pd.read_csv(ROOT / "data/confirmatory250_design.csv").sort_values("reading_order")
    vols = design.VolumeName.tolist(); w = design.design_weight.to_numpy()
    ref = (pd.read_csv(ROOT / "data/confirmatory250_reference_standard.csv")
           .pivot(index="volume_name", columns="label_name", values="consensus").loc[vols, LABELS]
           .replace(RULES["A"]).to_numpy(float))

    runs = {}
    for slug, name in MODELS.items():
        files = sorted(args.run_dir.glob(f"{slug}_run*.csv"))
        mats = []
        for f in files:
            d = pd.read_csv(f).set_index("VolumeName")
            if d.index.nunique() < len(vols) or d["_extraction_failed"].astype(bool).any():
                print(f"  注意: {f.name} は未完了または失敗あり（{int(d['_extraction_failed'].astype(bool).sum())} 件）")
            mats.append(d.reindex(vols)[LABELS].to_numpy(float))
        if mats:
            runs[name] = np.stack(mats)  # (runs, reports, labels)

    per_rows, sum_rows, run_rows = [], [], []
    for name, A in runs.items():
        k = A.shape[0]
        flip = (A.min(0) != A.max(0))                    # (reports, labels)
        sum_rows.append({"model": name, "runs": k,
                         "flip_rate": flip.mean(),
                         "flip_rate_weighted": (flip * w[:, None]).sum() / (w.sum() * len(LABELS)),
                         "fleiss_kappa": fleiss_kappa(A.reshape(k, -1).T),
                         "reports_with_any_flip": int(flip.any(1).sum())})
        for j, lab in enumerate(LABELS):
            per_rows.append({"model": name, "label": lab, "flip_rate": flip[:, j].mean(),
                             "flips": int(flip[:, j].sum()),
                             "fleiss_kappa": fleiss_kappa(A[:, :, j].T)})
        for r in range(k):
            _, macro, _, _ = summarize(ref, A[r], np.ones(len(vols)))
            run_rows.append({"model": name, "run": r + 1, "macro_f1": macro["f1"], "macro_kappa": macro["kappa"]})

    common = min((A.shape[0] for A in runs.values()), default=0)
    if len(runs) == 3 and common >= 2:
        MV = np.stack([(np.stack([runs[n][r] for n in runs]).sum(0) >= 2) for r in range(common)]).astype(float)
        flip = MV.min(0) != MV.max(0)
        sum_rows.append({"model": "Majority vote", "runs": common, "flip_rate": flip.mean(),
                         "flip_rate_weighted": (flip * w[:, None]).sum() / (w.sum() * len(LABELS)),
                         "fleiss_kappa": fleiss_kappa(MV.reshape(common, -1).T),
                         "reports_with_any_flip": int(flip.any(1).sum())})
        for r in range(common):
            _, macro, _, _ = summarize(ref, MV[r], np.ones(len(vols)))
            run_rows.append({"model": "Majority vote", "run": r + 1, "macro_f1": macro["f1"], "macro_kappa": macro["kappa"]})

    args.out.mkdir(exist_ok=True)
    S, P, R = pd.DataFrame(sum_rows), pd.DataFrame(per_rows), pd.DataFrame(run_rows)
    S.to_csv(args.out / "repeat_runs_summary.csv", index=False)
    P.to_csv(args.out / "repeat_runs_per_label.csv", index=False)
    R.to_csv(args.out / "repeat_runs_vs_reference.csv", index=False)
    print(S.to_string(index=False)); print(); print(R.to_string(index=False))
    t = ROOT / "logs/repeat_runs_summary.csv"
    if t.exists():
        print("\n■ 所要時間"); print(pd.read_csv(t)[["model", "run", "start_utc", "wall_s", "failed", "median_latency_s"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
