"""確認セットでのモデル間解析（PH-13）。API は使わない。

  - モデル間の一致（全インスタンス、Cohen κ）
  - 参照標準に対する誤りの相関（誤り指標どうしの phi 係数）と、2モデルが同時に誤る割合
  - leave-one-model-out: 残り2モデルの「両方陽性」（AND）と「どちらか陽性」（OR）の性能
"""
from __future__ import annotations
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_confirmatory import load, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
M3 = ["GPT-5.4", "GPT-5.4 mini", "DeepSeek V3.2"]


def main() -> int:
    vols, R, src, w, strata = load("A")
    ones = np.ones(len(vols)); rows = []
    for a, b in combinations(M3, 2):
        A, B = src[a].ravel(), src[b].ravel()
        ea, eb = (src[a] != R).ravel(), (src[b] != R).ravel()
        phi = np.corrcoef(ea, eb)[0, 1]
        rows.append({"pair": f"{a} / {b}", "agreement": (A == B).mean(), "kappa": cohen_kappa_score(A, B),
                     "error_phi": phi, "both_wrong": int((ea & eb).sum()),
                     "a_errors": int(ea.sum()), "b_errors": int(eb.sum()),
                     "both_wrong_given_a_wrong": (ea & eb).sum() / ea.sum()})
    T = pd.DataFrame(rows); T.to_csv(ROOT / "tables/confirmatory_intermodel.csv", index=False)
    print("■ モデル間一致と誤り相関")
    for _, r in T.iterrows():
        print(f"  {r.pair:28s} 一致 {r.agreement*100:5.2f}% κ {r.kappa:.3f} | 誤り phi {r.error_phi:.3f} "
              f"同時誤り {r.both_wrong} (誤り {r.a_errors}/{r.b_errors}, P(b誤|a誤)={r.both_wrong_given_a_wrong:.2f})")
    lo = []
    for out in M3:
        keep = [m for m in M3 if m != out]
        AND = (src[keep[0]] * src[keep[1]]).astype(float); OR = ((src[keep[0]] + src[keep[1]]) > 0).astype(float)
        for nm, S in (("AND", AND), ("OR", OR)):
            _, m, _, _ = summarize(R, S, ones)
            lo.append({"left_out": out, "rule": nm, "macro_f1": m["f1"], "macro_kappa": m["kappa"],
                       "sens": m["sensitivity"], "spec": m["specificity"]})
    L = pd.DataFrame(lo); L.to_csv(ROOT / "tables/confirmatory_leave_one_out.csv", index=False)
    _, mv, _, _ = summarize(R, src["Majority vote"], ones)
    print(f"\n■ leave-one-model-out（参考: 3モデル多数決 F1 {mv['f1']*100:.1f} κ {mv['kappa']:.3f}）")
    for _, r in L.iterrows():
        print(f"  除外 {r.left_out:14s} {r.rule:3s} F1 {r.macro_f1*100:5.1f} κ {r.macro_kappa:.3f} sens {r.sens*100:5.1f} spec {r.spec*100:5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
