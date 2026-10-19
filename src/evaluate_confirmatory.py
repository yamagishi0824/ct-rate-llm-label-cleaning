"""confirmatory250 参照標準に対するラベルソース別の性能（PH-01）。

参照標準: data/confirmatory250_reference_standard.csv の合議値を Rule A で二値化
          （present / uncertain を陽性）。
ラベルソース: CT-RATE 元ラベル、GPT-5.4、GPT-5.4 mini、DeepSeek V3.2、
              3モデル多数決（2/3 以上が陽性）。
指標: カテゴリ別の sensitivity / specificity / PPV / NPV / accuracy / F1 / Cohen κ、
      label-macro 平均と micro 平均。非重み付けと層別デザイン重み付けの両方。
信頼区間: report 単位 bootstrap 2,000 回。層別抽出なので層内で復元抽出する。
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]
MODELS = {"GPT-5.4": "gpt-5.4", "GPT-5.4 mini": "gpt-5.4-mini", "DeepSeek V3.2": "DeepSeek-V3.2"}
RULES = {
    "A": {"present": 1, "uncertain": 1, "negated": 0, "not_mentioned": 0},
    "B": {"present": 1, "uncertain": 0, "negated": 0, "not_mentioned": 0},
}
METRICS = ["sensitivity", "specificity", "ppv", "npv", "accuracy", "f1", "kappa"]


def load(rule: str):
    design = pd.read_csv(ROOT / "data/confirmatory250_design.csv").sort_values("reading_order")
    vols = design["VolumeName"].tolist()
    ref = pd.read_csv(ROOT / "data/confirmatory250_reference_standard.csv")
    ref = ref.pivot(index="volume_name", columns="label_name", values="consensus").loc[vols, LABELS]
    R = ref.replace(RULES[rule]).to_numpy(dtype=float)

    sources = {}
    ct = pd.read_csv(ROOT / "data/CT-RATE_reports.csv").set_index("VolumeName")
    sources["CT-RATE"] = ct.loc[vols, LABELS].to_numpy(dtype=float)
    for name, slug in MODELS.items():
        df = pd.read_csv(ROOT / f"data/CT-RATE_reports_{slug}.csv").set_index("VolumeName")
        sources[name] = df.loc[vols, LABELS].to_numpy(dtype=float)
    stack = np.stack([sources[n] for n in MODELS])
    sources["Majority vote"] = (stack.sum(axis=0) >= 2).astype(float)
    w = design["design_weight"].to_numpy(dtype=float)
    strata = design["stratum"].to_numpy()
    return vols, R, sources, w, strata


def counts(R, S, w):
    """重み付き混同行列（ラベル別）。w は report ごとの重み。"""
    w = w[:, None]
    tp = (w * (S == 1) * (R == 1)).sum(0)
    fp = (w * (S == 1) * (R == 0)).sum(0)
    fn = (w * (S == 0) * (R == 1)).sum(0)
    tn = (w * (S == 0) * (R == 0)).sum(0)
    return tp, fp, fn, tn


def metrics(tp, fp, fn, tn):
    with np.errstate(divide="ignore", invalid="ignore"):
        n = tp + fp + fn + tn
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        acc = (tp + tn) / n
        f1 = 2 * tp / (2 * tp + fp + fn)
        po = acc
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
        kappa = (po - pe) / (1 - pe)
    return dict(sensitivity=sens, specificity=spec, ppv=ppv, npv=npv,
                accuracy=acc, f1=f1, kappa=kappa)


def summarize(R, S, w):
    tp, fp, fn, tn = counts(R, S, w)
    per = metrics(tp, fp, fn, tn)
    macro = {k: np.nanmean(v) for k, v in per.items()}
    micro = {k: float(v) for k, v in metrics(tp.sum(), fp.sum(), fn.sum(), tn.sum()).items()}
    return per, macro, micro, (tp, fp, fn, tn)


def bootstrap(R, sources, w, strata, weighted: bool, B: int, seed: int):
    rng = np.random.default_rng(seed)
    idx_by_stratum = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    out = {name: {f"macro_{m}": [] for m in METRICS} | {f"micro_{m}": [] for m in METRICS}
           for name in sources}
    for _ in range(B):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_by_stratum])
        ww = w[idx] if weighted else np.ones(len(idx))
        for name, S in sources.items():
            _, macro, micro, _ = summarize(R[idx], S[idx], ww)
            for m in METRICS:
                out[name][f"macro_{m}"].append(macro[m])
                out[name][f"micro_{m}"].append(micro[m])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", default="A", choices=list(RULES))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--out", type=Path, default=ROOT / "tables")
    args = ap.parse_args()

    vols, R, sources, w, strata = load(args.rule)
    kish = w.sum() ** 2 / (w ** 2).sum()
    print(f"参照標準: {len(vols)} reports x {len(LABELS)} labels / Rule {args.rule}")
    print(f"Kish 有効標本サイズ: {kish:.1f}\n")

    args.out.mkdir(exist_ok=True)
    per_rows, summary_rows = [], []
    for weighted in (False, True):
        tag = "weighted" if weighted else "unweighted"
        ww = w if weighted else np.ones(len(vols))
        boot = bootstrap(R, sources, w, strata, weighted, args.boot, args.seed)
        for name, S in sources.items():
            per, macro, micro, (tp, fp, fn, tn) = summarize(R, S, ww)
            for i, lab in enumerate(LABELS):
                per_rows.append({"weighting": tag, "source": name, "label": lab,
                                 "TP": tp[i], "FP": fp[i], "FN": fn[i], "TN": tn[i],
                                 **{m: per[m][i] for m in METRICS}})
            row = {"weighting": tag, "source": name}
            for m in METRICS:
                for agg, val in (("macro", macro[m]), ("micro", micro[m])):
                    lo, hi = np.nanpercentile(boot[name][f"{agg}_{m}"], [2.5, 97.5])
                    row[f"{agg}_{m}"] = val
                    row[f"{agg}_{m}_lo"] = lo
                    row[f"{agg}_{m}_hi"] = hi
            summary_rows.append(row)

    per_df = pd.DataFrame(per_rows)
    sum_df = pd.DataFrame(summary_rows)
    per_df.to_csv(args.out / f"confirmatory_per_label_rule{args.rule}.csv", index=False)
    sum_df.to_csv(args.out / f"confirmatory_summary_rule{args.rule}.csv", index=False)

    for tag in ("unweighted", "weighted"):
        print(f"=== {tag} / label-macro ===")
        s = sum_df[sum_df.weighting == tag]
        for _, r in s.iterrows():
            print(f"  {r.source:14s} F1 {r.macro_f1*100:5.1f} ({r.macro_f1_lo*100:4.1f}-{r.macro_f1_hi*100:4.1f})"
                  f"  k {r.macro_kappa:.3f} ({r.macro_kappa_lo:.3f}-{r.macro_kappa_hi:.3f})"
                  f"  sens {r.macro_sensitivity*100:5.1f}  spec {r.macro_specificity*100:5.1f}")
        print()
    print(f"出力: {args.out}/confirmatory_summary_rule{args.rule}.csv, confirmatory_per_label_rule{args.rule}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
