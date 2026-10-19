"""確認セットの図（Figure 4、Supplementary Figure 2）。

Figure 4 は開発セットの Figure 3（notebooks/model_vs_human.ipynb の
mvh_fig1_2_combined_heatmap.png）と同じレイアウト: 上段 F1、下段 Cohen κ のヒートマップ、
行は CT-RATE の値で降順、RdYlGn、0–100%。
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_confirmatory import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.titlesize": 12,
                     "axes.labelsize": 11, "legend.fontsize": 10, "figure.dpi": 150})
NAME = {"CT-RATE": "CT-RATE (original)", "GPT-5.4": "GPT-5.4", "GPT-5.4 mini": "GPT-5.4 mini",
        "DeepSeek V3.2": "DeepSeek-V3.2", "Majority vote": "Majority vote (3 LLMs)"}
ORDER = list(NAME.values())


def figure4():
    d = pd.read_csv(ROOT / "tables/confirmatory_per_label_full_ruleA.csv")
    d["Source"] = d.Source.map(NAME)
    fig, axes = plt.subplots(2, 1, figsize=(len(ORDER) * 2.4 + 2, 14))
    for ax, metric, title in zip(axes, ["F1", "Kappa"],
                                 ["F1 score vs confirmatory reference standard",
                                  "Cohen's κ vs confirmatory reference standard"]):
        p = d.pivot(index="Label", columns="Source", values=metric)[ORDER]
        p = p.loc[p["CT-RATE (original)"].sort_values(ascending=False).index]
        annot = p.map(lambda v: f"{v * 100:.1f}%" if not np.isnan(v) else "")
        sns.heatmap(p.astype(float), annot=annot, fmt="", cmap="RdYlGn", vmin=0, vmax=1,
                    linewidths=0.4, linecolor="white", annot_kws={"size": 9},
                    cbar_kws={"label": metric, "shrink": 0.7, "format": lambda x, _: f"{x*100:.0f}%"}, ax=ax)
        ax.set_title(title, pad=10)
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
        ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout()
    out = FIG / "conf_fig4_combined_heatmap.png"
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
    return out


def sfigure2():
    vols, R, src, w, strata = load("A")
    keys = ["CT-RATE", "GPT-5.4", "GPT-5.4 mini", "DeepSeek V3.2"]
    E = np.stack([(src[k] != R).ravel().astype(float) for k in keys])
    phi = pd.DataFrame(np.corrcoef(E), index=[NAME[k] for k in keys], columns=[NAME[k] for k in keys])
    mask = np.triu(np.ones_like(phi, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(phi, mask=mask, annot=phi.map(lambda v: f"{v:.3f}"), fmt="", cmap="Blues", vmin=0, vmax=1,
                linewidths=0.4, linecolor="white", annot_kws={"size": 10}, square=True,
                cbar_kws={"label": "phi", "shrink": 0.7}, ax=ax)
    ax.set_title("Error correlation vs confirmatory reference standard", pad=10)
    ax.tick_params(axis="x", rotation=20, labelsize=9); ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    out = FIG / "conf_sfig2_error_phi_heatmap.png"
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
    return out, phi


if __name__ == "__main__":
    print(figure4())
    o, phi = sfigure2(); print(o); print(phi.round(3).to_string())
