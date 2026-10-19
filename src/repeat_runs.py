"""confirmatory250 に対する反復ラン（PH-09 / PH-10）。

各モデルを confirmatory250 の 250 レポートに --runs 回適用する。プロンプトと設定は
本番抽出（src/extract_labels.py）と同一。temperature / top-p / seed はサービス既定値。

出力
  data/repeat_runs/<model>_run<k>.csv   ラベル 18 列 + 生応答 + 失敗フラグ + 時刻 + 試行回数
  logs/repeat_runs_<model>.jsonl        リクエスト単位のログ（追記）
  logs/repeat_runs_summary.csv          ラン単位の開始・終了時刻、所要時間、失敗件数（追記）

例
  export AZURE_AI_ENDPOINT=... AZURE_AI_KEY=...
  python src/repeat_runs.py --model-name gpt-5.4 --runs 3
  python src/repeat_runs.py --model-name gpt-5.4 --runs 1 --limit 2   # 動作確認
途中で止まっても同じコマンドで再開できる（完了済みの行は飛ばす）。
"""
from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_labels import (LABEL_COLS, _clean, _default_labels, _from_key,  # noqa: E402
                            _model_slug, call_llm, init_client)
from extract_labels_4level import call_llm_4  # noqa: E402

SCHEMES = {
    "binary": (call_llm, _default_labels),
    "4level": (call_llm_4, lambda: {k: None for k in _from_key}),
}

ROOT = Path(__file__).resolve().parents[1]
now = lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def run_once(client, model_name, rows, out_path, log_path, workers, retries, run_k, scheme):
    call, default = SCHEMES[scheme]
    done = {}
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = {r.VolumeName: r for r in prev.itertuples(index=False) if not r._extraction_failed}
    todo = [r for r in rows if r["VolumeName"] not in done]

    def one(row):
        f, i = _clean(row["Findings_EN"]), _clean(row["Impressions_EN"])
        rec = {"VolumeName": row["VolumeName"], "run": run_k, "model": model_name, "scheme": scheme,
               "start_utc": now(), "attempts": 0, "_parse_error": "", "_raw_response": ""}
        t0 = time.monotonic()
        labels = None
        for a in range(1, retries + 1):
            rec["attempts"] = a
            try:
                labels, diag = call(client=client, model_name=model_name,
                                        findings=f, impression=i)
                rec["_raw_response"] = diag.get("_raw_response") or ""
                rec["_parse_error"] = diag.get("_parse_error", "")
            except Exception as e:  # API エラーは再試行
                rec["_parse_error"] = f"api_error: {type(e).__name__}: {e}"
            if labels is not None:
                break
            time.sleep(min(2 ** a, 20))
        rec["end_utc"] = now()
        rec["latency_s"] = round(time.monotonic() - t0, 3)
        rec["_extraction_failed"] = labels is None
        labels = labels or default()
        for key, col in _from_key.items():
            rec[col] = labels[key]
        return rec

    t_start, wall0 = now(), time.monotonic()
    recs = []
    with ThreadPoolExecutor(max_workers=workers) as ex, open(log_path, "a") as log:
        futs = [ex.submit(one, r) for r in todo]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{model_name} run{run_k}"):
            rec = fut.result()
            recs.append(rec)
            log.write(json.dumps({k: rec[k] for k in rec if k != "_raw_response"}, ensure_ascii=False) + "\n")
            log.flush()
    wall = time.monotonic() - wall0

    new = pd.DataFrame(recs)
    old = pd.DataFrame([r._asdict() for r in done.values()]) if done else pd.DataFrame()
    out = pd.concat([old, new], ignore_index=True)
    order = {r["VolumeName"]: k for k, r in enumerate(rows)}
    out = out.sort_values("VolumeName", key=lambda s: s.map(order))
    cols = ["VolumeName", "model", "scheme", "run", *LABEL_COLS, "_extraction_failed", "_parse_error",
            "attempts", "start_utc", "end_utc", "latency_s", "_raw_response"]
    out[cols].to_csv(out_path, index=False)
    return {"model": model_name, "scheme": scheme, "run": run_k, "start_utc": t_start, "end_utc": now(),
            "wall_s": round(wall, 1), "requested": len(todo),
            "failed": int(new["_extraction_failed"].sum()) if len(new) else 0,
            "median_latency_s": float(new["latency_s"].median()) if len(new) else None,
            "workers": workers}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--scheme", choices=list(SCHEMES), default="binary",
                    help="binary = 本番と同じ二値 / 4level = present・uncertain・negated・not_mentioned")
    ap.add_argument("--start-run", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="先頭 N 本だけ（動作確認用）")
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    client, model_name = init_client(args.model_name, args.endpoint, args.api_key)
    slug = _model_slug(args.model_name)
    vols = pd.read_csv(ROOT / "data/confirmatory250_design.csv").sort_values("reading_order").VolumeName
    rep = pd.read_csv(ROOT / "data/CT-RATE_reports.csv").set_index("VolumeName")
    rows = [{"VolumeName": v, "Findings_EN": rep.at[v, "Findings_EN"],
             "Impressions_EN": rep.at[v, "Impressions_EN"]} for v in vols]
    if args.limit:
        rows = rows[: args.limit]

    out_dir = ROOT / "data/repeat_runs"; out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "logs"; log_dir.mkdir(exist_ok=True)
    summ_path = log_dir / "repeat_runs_summary.csv"
    for k in range(args.start_run, args.start_run + args.runs):
        tag = "" if args.scheme == "binary" else "_4level"
        out_path = out_dir / f"{slug}{tag}_run{k}.csv"
        s = run_once(client, model_name, rows, out_path, log_dir / f"repeat_runs_{slug}{tag}.jsonl",
                     args.num_workers, args.max_retries, k, args.scheme)
        pd.DataFrame([s]).to_csv(summ_path, mode="a", header=not summ_path.exists(), index=False)
        print(f"  run{k}: {s['wall_s']} s / 失敗 {s['failed']} / 中央値 {s['median_latency_s']} s → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
