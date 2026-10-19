"""confirmatory250 アノテーションの事前入力。

読影者自身の既読レポートから判断を再適用する。他読影者の情報は一切使わないため、
読影者間の独立性は損なわれない。

  段階2  文集合一致 : あるラベルに関係する文の集合が既読レポートと完全一致し、
                      かつ過去の判定が一意なら、その値を入れる。
  段階3  語句ルール : 否定も hedge も伴わない明確な陽性語句があれば present を入れる。
                      対象は修飾語になりにくい 6 ラベルに限定。

「言及なし → not_mentioned」はツール側の一括ボタンで処理済みのため、ここでは扱わない。

入力済みの値は上書きしない。前回この스크립트が入れた値（auto フラグ付き）のみ、
辞書が育った場合に更新する。
"""
from __future__ import annotations
import argparse, json, re, shutil, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "CT-RATE_reports.csv"
DESIGN = ROOT / "data" / "confirmatory250_design.csv"

LABELS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]

# 段階2: 文集合を作るための「そのラベルに言及しうる語」
TRIGGER = {
    "Medical material": r"catheter|stent|pacemaker|port\b|tube|cannula|material|prosthe|valve|wire|clip|drain|suture|device|graft|electrode|implant|tracheostom",
    "Arterial wall calcification": r"atherom|atheroscler|plaque|calcific|calcifi",
    "Cardiomegaly": r"cardiomegal|heart|cardiac",
    "Pericardial effusion": r"pericard",
    "Coronary artery wall calcification": r"coronar|rca\b|lad\b|circumflex",
    "Hiatal hernia": r"hiatal|hiatus|hernia",
    "Lymphadenopathy": r"lymph|lap\b|node",
    "Emphysema": r"emphysem|bulla|air trapping",
    "Atelectasis": r"atelecta|collaps|non.?aerated|aeration",
    "Lung nodule": r"nodul|mass",
    "Lung opacity": r"ground.?glass|opacit|densit|hazy|infiltrat|tree|bud|crazy|consolidat",
    "Pulmonary fibrotic sequela": r"fibro|sequela|sequelae|scar|honeycomb|reticul|traction|retraction|distortion",
    "Pleural effusion": r"pleural|effusion|hemithorax|fluid",
    "Mosaic attenuation pattern": r"mosaic|attenuation difference|perfusion",
    "Peribronchial thickening": r"peribronch|bronchial wall|cuffing",
    "Consolidation": r"consolidat|infiltrat|air bronchogram",
    "Bronchiectasis": r"bronchiectas|bronchial dilat|dilated bronch",
    "Interlobular septal thickening": r"septal|septa\b|kerley|crazy|reticul",
}

# 段階3: 否定も hedge も無ければ present と断定してよい語句。
# 値は正規表現、または {pat, require, exclude, veto} の辞書。
#   require … 同じ文に必要な語     exclude … 同じ文にあれば発火しない語
#   veto    … レポート全体にあれば発火しない語
POSITIVE = {
    "Atelectasis": r"atelecta",
    "Emphysema": r"emphysemat|\bemphysema\b",
    "Mosaic attenuation pattern": r"mosaic",
    "Hiatal hernia": r"hiatal hernia|hiatus hernia",
    "Peribronchial thickening": r"peribronchial thicken|bronchial wall thicken|peribronchial cuff",
    "Interlobular septal thickening": r"interlobular septal thicken|septal line|kerley",
    # 動脈壁の石灰化。石灰化語は atheroscler / atherom / plaque の直前2語以内に限る。
    # "calcified pleural plaques"（胸膜）と
    # "coronary arteries in the descending aorta"（冠動脈の位置を述べただけ）は除外。
    "Arterial wall calcification": {
        "pat": r"(?:calcifi\\w*\\s+(?:\\w+\\s+){0,2}(?:atheroscler\\w*|atherom\\w*|plaques?)|atheroscler\\w*\\s+wall\\s+calcifi\\w*)[^.]{0,80}\baort",
        "exclude": r"pleural plaque|coronary arter\w*\s+in the\s+\w*\s*aorta",
    },
    "Coronary artery wall calcification": {
        "pat": r"(?:calcifi\\w*\\s+(?:\\w+\\s+){0,2}(?:atheroscler\\w*|atherom\\w*|plaques?)|atheroscler\\w*\\s+wall\\s+calcifi\\w*)[^.]{0,80}(?:coronar|\brca\b|\blad\b)",
        "exclude": r"pleural plaque",
    },
    # 肺結節。tree-in-bud / すりガラス結節 / 瘢痕性石灰化結節 / 他臓器の結節は該当しない
    "Lung nodule": {
        "pat": r"\bnodules?\b|\bnodular lesion",
        "require": r"lung|pulmonar",
        "exclude": (r"centriacinar|tree|bud|ground.?glass|sequela|calcifi|thyroid|adrenal|"
                    r"liver|hepat|kidney|renal|spleen|schmorl|vertebra|bone|breast|lymph|"
                    r"pancrea|fissure|nodular (?:consolidat|densit|infiltrat|opacit|structure)"),
        "veto": r"nodular consolidation\?|\(nodular",
    },
}

# 段階3b: この語句があり、かつレポート全体に打ち消す記載が無ければ negated と断定してよい
# (label -> (発火する語句, レポート全体での除外語))
NEGATIVE = {
    "Cardiomegaly": (
        r"heart (?:contour|size|dimension)s?,? ?(?:size|width)?s? (?:are|is|was|were) (?:normal|natural)|"
        r"heart (?:contour|size|dimension)[^.]*\b(?:normal|natural)\b|"
        r"\bcto\b[^.]*\b(?:normal|natural|within)\b|"
        r"cardiothoracic (?:ratio|index)[^.]*\b(?:normal|natural|within)\b",
        r"cardiomegal|heart (?:size|dimension)s?[^.]*increas|increase[^.]*heart size|"
        r"enlarge\w*[^.]*heart|heart[^.]*enlarge",
    ),
}

NEG = re.compile(r"\b(no|not|without|absence)\b")
HEDGE = re.compile(r"\?|suspic|suspect|possib|cannot|may be|in favor of|consider|probab|compatib|evaluat")
# 胸部以外の臓器の記述。肺・胸郭ラベルの判断材料から外す
OFF_ORGAN = re.compile(
    r"liver|hepat|adrenal|kidney|renal|spleen|thyroid|vertebra|bone|osseous|corpus|"
    r"abdomen|abdominal|gallbladder|pancrea|breast|schmorl|scoliosis|kyphosis"
)
THORACIC_ONLY = set(LABELS) - {"Hiatal hernia", "Medical material"}


def load_text() -> dict[str, list[str]]:
    df = pd.read_csv(REPORTS)
    text = (df["Findings_EN"].fillna("") + " || " + df["Impressions_EN"].fillna("")).str.lower()
    return {
        v: [re.sub(r"\s+", " ", s).strip() for s in re.split(r"(?<=\.)\s+", t)]
        for v, t in zip(df["VolumeName"], text)
    }


def relevant(sents: list[str], label: str) -> list[str]:
    """label の判断材料になる文だけを返す。"""
    pat = TRIGGER[label]
    out = []
    for s in sents:
        if not re.search(pat, s):
            continue
        if label in THORACIC_ONLY and OFF_ORGAN.search(s):
            continue
        out.append(s)
    return out


def sentence_key(sents: list[str], label: str) -> tuple[str, ...]:
    return tuple(sorted(relevant(sents, label)))


def positive_fires(sents: list[str], label: str) -> bool:
    spec = POSITIVE[label]
    if isinstance(spec, str):
        spec = {"pat": spec}
    veto = spec.get("veto")
    if veto and re.search(veto, " ".join(sents)):
        return False
    require, exclude = spec.get("require"), spec.get("exclude")
    for s in sents:
        if not re.search(spec["pat"], s):
            continue
        if require and not re.search(require, s):
            continue
        if exclude and re.search(exclude, s):
            continue
        if label in THORACIC_ONLY and OFF_ORGAN.search(s):
            continue
        if NEG.search(s) or HEDGE.search(s):
            continue
        return True
    return False


def negative_fires(sents: list[str], label: str) -> bool:
    """否定ルールが成立するか。レポート全体に打ち消す記載があれば発火しない。"""
    pattern, veto = NEGATIVE[label]
    joined = " ".join(sents)
    if re.search(veto, joined):
        return False
    return any(re.search(pattern, s) for s in sents)


def is_human(rec: dict | None) -> bool:
    """読影者が自分で入れた値か（auto フラグが無く、値がある）。"""
    return bool(rec) and bool(rec.get("value")) and not rec.get("auto")


def build_dictionary(done: dict, texts: dict) -> tuple[dict, list]:
    """既読から (ラベル, 文集合) -> 値 の辞書を作る。判定が割れたものは除外して返す。"""
    seen = defaultdict(set)
    for vol, recs in done.items():
        for label in LABELS:
            key = sentence_key(texts[vol], label)
            if key:
                seen[(label, key)].add(recs[label]["value"])
    unique = {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}
    conflicts = [(k, sorted(v)) for k, v in seen.items() if len(v) > 1]
    return unique, conflicts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("state", type=Path, help="annotation_state_<読影者>.json")
    ap.add_argument("--apply", action="store_true", help="書き込む（既定は dry-run）")
    args = ap.parse_args()

    texts = load_text()
    targets = list(pd.read_csv(DESIGN)["VolumeName"])
    state = json.loads(args.state.read_text(encoding="utf-8"))
    data = state.setdefault("data", {})
    done = {v: r for v, r in data.items() if len(r) == 18}

    print(f"読影者: {state.get('annotator') or '(未設定)'}")
    print(f"既読 {len(done)} 本 / 対象 {len(targets)} 本")

    unique, conflicts = build_dictionary(done, texts)
    print(f"文集合パターン {len(unique) + len(conflicts)} 種（一意 {len(unique)} / 判定が割れている {len(conflicts)}）")
    if conflicts:
        print("\n  同じ文なのに判定が異なるパターン（先に統一を推奨）:")
        for (label, key), vals in conflicts:
            print(f"    [{label}] {vals}")
            print(f"       {key[0][:96]}")
        print()

    # 既読に対して語句ルールを当て、present になっていないものを警告
    suspect = []
    for vol, recs in done.items():
        for label in POSITIVE:
            if positive_fires(texts[vol], label) and recs[label]["value"] != "present":
                suspect.append((vol, label, recs[label]["value"], "present"))
        for label in NEGATIVE:
            if negative_fires(texts[vol], label) and recs[label]["value"] != "negated":
                suspect.append((vol, label, recs[label]["value"], "negated"))
    if suspect:
        print(f"  既読 {len(done)} 本で語句ルールと食い違う箇所 {len(suspect)} 件（要確認）:")
        for vol, label, val, want in suspect:
            print(f"    {vol.replace('_a_1.nii.gz', '')}  {label} = {val}  → {want} の可能性")
        print()

    added = defaultdict(int)
    refreshed = kept = 0
    for vol in targets:
        if vol in done:
            continue
        recs = data.setdefault(vol, {})
        for label in LABELS:
            cur = recs.get(label)
            if is_human(cur):
                kept += 1
                continue
            value = None
            if label in POSITIVE and positive_fires(texts[vol], label):
                value, stage = "present", "語句(陽性)"
            elif label in NEGATIVE and negative_fires(texts[vol], label):
                value, stage = "negated", "語句(陰性)"
            else:
                key = sentence_key(texts[vol], label)
                if key and (label, key) in unique:
                    value, stage = unique[(label, key)], "文集合"
            if value is None:
                continue
            if cur and cur.get("value") == value:
                refreshed += 1
                continue
            recs[label] = {
                "value": value,
                "evidence": "[auto]" if value != "not_mentioned" else "",
                "temporality": "current" if value in ("present", "uncertain") else "",
                "auto": True,
            }
            added[(stage, label)] += 1

    total = sum(added.values())
    print(f"事前入力: 新規・変更 {total} 件 / 据え置き {refreshed} 件 / 手入力を保護 {kept} 件\n")
    for stage in ("文集合", "語句(陽性)", "語句(陰性)"):
        rows = {l: n for (s, l), n in added.items() if s == stage}
        if rows:
            print(f"  [{stage}一致] 計 {sum(rows.values())} 件")
            for label, n in sorted(rows.items(), key=lambda x: -x[1]):
                print(f"      {label:34s} {n:4d}")

    if not args.apply:
        print("\ndry-run。書き込むには --apply を付けてください。")
        return 0

    backup = args.state.with_name(
        f"{args.state.stem}.bak_{datetime.now():%Y%m%d_%H%M%S}{args.state.suffix}")
    shutil.copy2(args.state, backup)
    args.state.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n書き込み完了: {args.state}")
    print(f"バックアップ: {backup}")
    print("ツールで「作業状態を読み込み」から読み直してください。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
