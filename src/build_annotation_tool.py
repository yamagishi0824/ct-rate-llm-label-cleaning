#!/usr/bin/env python3
"""build_annotation_tool.py

confirmatory reference set（250レポート）用のアノテーションツール（単一 HTML）を生成する。

- 依存なし・オフライン動作。生成された HTML をダブルクリックでブラウザが開く
- レポート本文は HTML に埋め込むので file:// でも動く
- 入力は自動で localStorage に保存され、CSV / JSON で書き出せる
- CT-RATE ラベル・LLM ラベル・層の割り当ては一切表示しない（blinded）
- 日本語訳（data/confirmatory250_reports_ja.json）が存在すれば対訳ペインに表示する

Usage
-----
  python src/build_annotation_tool.py
  open annotation_tool/annotate.html
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "annotation_tool"

LABELS = [
    ("Medical material", "医療材料", "撮像範囲内の医療デバイス・異物。ペースメーカー、CVライン、ポート、ステント、PICC、気管チューブ、経鼻胃管、人工弁、外科クリップ／ステープル、術後の高吸収物質、異物。骨の硬化性変化、腫瘍性の implants（播種・転移巣）は該当しない。"),
    ("Arterial wall calcification", "動脈壁石灰化", "大動脈・大動脈弓・主要胸部動脈（範囲内の腹部大動脈を含む）の石灰化が明示されている場合。atheroma plaque のみ、atherosclerotic changes のみは該当しない。冠動脈のみ、弁・心膜の石灰化も該当しない。"),
    ("Cardiomegaly", "心拡大", "cardiomegaly、心陰影拡大、個別の心腔拡大。"),
    ("Pericardial effusion", "心嚢液", "心膜液・心嚢液。stable と書かれていても現在存在するので該当。"),
    ("Coronary artery wall calcification", "冠動脈壁石灰化", "冠動脈またはその分枝（LAD 等）の石灰化が明示されている場合。石灰化非明示の atheroma plaque、大動脈弁石灰化は該当しない。"),
    ("Hiatal hernia", "横隔膜（食道）裂孔ヘルニア", "hiatal / hiatus hernia、横隔膜を通した胃・腸の嵌入。"),
    ("Lymphadenopathy", "リンパ節腫大", "病的腫大の明示（lymphadenopathy / LAP / lymphadenomegaly / pathological size）、または短径 10mm 以上。撮像範囲内のどの station でも可。millimetric・subcentimeter・「病的サイズに達しない」は【否】。石灰化のみで腫大なしも【否】。腫大が縮小しても現存すれば【有】。節性病変の疑い（lymph node? thymic mass?）は【疑】。"),
    ("Emphysema", "肺気腫", "emphysema / emphysematous change（亜型は問わない）、薄壁 bulla の形成。air trapping 単独は該当しない。"),
    ("Atelectasis", "無気肺", "atelectasis / collapse（subsegmental, linear, plate-like, compressive を含む）、fibroatelectatic change、無含気の記述。"),
    ("Lung nodule", "肺結節", "個別の肺結節（solid / part-solid / GGN）。mass でもサイズが結節範囲なら該当。びまん性・patchy な nodular ground-glass、tree-in-bud の centriacinar nodularity、線維性瘢痕の nodular structure は該当しない。"),
    ("Lung opacity", "肺野の濃度上昇", "GGO、hazy / 非特異的な濃度上昇、dependent density、tree-in-bud、crazy paving、肺炎性の infiltration。無気肺と明示的に解釈されている濃度上昇、瘢痕（sequela）と解釈されているものは該当しない。"),
    ("Pulmonary fibrotic sequela", "肺の線維性瘢痕", "炎症後の線維性瘢痕。fibrotic / fibroatelectatic change、容積減少を伴う線状瘢痕、traction bronchiectasis を伴う構築の歪み、pleuroparenchymal retraction。間質性肺炎パターン（honeycombing、活動性 IP の reticulation）、石灰化のみの sequela、胸膜単独の肥厚は該当しない。"),
    ("Pleural effusion", "胸水", "片側／両側の胸水（minimal を含む）。"),
    ("Mosaic attenuation pattern", "モザイク様吸収値", "mosaic attenuation / mosaic perfusion / mosaic density differences が明示されている場合。"),
    ("Peribronchial thickening", "気管支壁肥厚", "気管支壁肥厚、peribronchial cuffing。peribronchial が他の所見の分布を述べているだけの場合は該当しない。"),
    ("Consolidation", "浸潤影", "airspace consolidation、bronchopneumonic infiltration、air bronchogram を伴う濃度上昇。ground-glass consolidation は Consolidation と Lung opacity の両方。"),
    ("Bronchiectasis", "気管支拡張", "bronchiectasis（型は問わない）、気管支の拡張の記述。"),
    ("Interlobular septal thickening", "小葉間隔壁肥厚", "小葉間隔壁の肥厚、septal line、Kerley B line、crazy paving（Lung opacity も併せて該当）。小葉間隔壁肥厚が明示されない IP 疑いのみは該当しない。"),
]

# ── マーカー用キーワード（正規表現、大文字小文字無視） ──────────────────────
# 目的は「見落とし防止」なので再現率優先。偽陽性の語が光っても読影者が判断する。
KEYWORDS: dict[str, list[str]] = {
    "Medical material": [r"pacemaker", r"pes ?maker", r"catheter", r"\bport\b", r"\bstent", r"picc",
        r"endotracheal", r"nasogastric", r"prosthe", r"valve replacement", r"\bclips?\b", r"staple",
        r"suture material", r"metallic", r"foreign bod", r"\bimplant", r"cannula", r"shrapnel",
        r"sternotomy", r"embolizing", r"hyperdense material", r"pace ?maker",
        r"tracheosto", r"tracheostem", r"surgical material", r"fixator", r"cerclage",
        r"valvuloplasty", r"screw", r"\bmesh\b", r"\bgraft\b", r"\bcoil\b", r"\bwire", r"\btube\b",
        r"\bdrain", r"\blead\b", r"\bleads\b", r"electrode", r"\bmaterial\b"],
    "Arterial wall calcification": [r"calcific", r"calcifi", r"atheroma", r"atheromat",
        r"atherosclerot", r"\baorta\b", r"aortic", r"aortic arch", r"brachiocephalic", r"subclavian"],
    "Cardiomegaly": [r"cardiomegal", r"cardiac silhouette", r"\bheart\b", r"cardiac",
        r"\bcto\b", r"cardiothoracic", r"atrium", r"atrial", r"ventric",
        r"dilat.{0,20}heart", r"heart.{0,20}enlarg", r"enlarg.{0,20}heart"],
    "Pericardial effusion": [r"pericardi"],
    "Coronary artery wall calcification": [r"coronary", r"\blad\b", r"\brca\b", r"\blcx\b",
        r"circumflex", r"left main"],
    "Hiatal hernia": [r"hiatal", r"hiatus", r"herniat", r"\bhernia\b"],
    "Lymphadenopathy": [r"lymph node", r"lymphadenopath", r"\blap\b", r"lymphadenomegal",
        r"lymphadenit", r"\bnodal\b", r"paratracheal", r"prevascular", r"precarinal", r"subcarinal",
        r"aortopulmonary", r"aorticopulmonary", r"hilar", r"axillar", r"supraclavicular",
        r"millimetric", r"pathological (size|dimension|configuration)", r"short ax", r"fatty hilus"],
    "Emphysema": [r"emphysema", r"emphysemat", r"bulla", r"bullous", r"air trapping",
        r"centrilobular", r"paraseptal", r"panlobular", r"air cyst"],
    "Atelectasis": [r"atelect", r"ateletat", r"atelekt", r"collaps", r"not ventilated",
        r"non-?aerated", r"loss of aeration", r"aeration", r"volume loss", r"fibroatelect",
        r"fibrotatelect"],
    "Lung nodule": [r"nodul", r"\bmass\b", r"masses", r"metasta", r"\bggn\b", r"micronodul",
        r"spicul", r"\blesion\b"],
    "Lung opacity": [r"ground ?-?glass", r"opacit", r"density increase", r"increase in density",
        r"densit", r"hazy", r"infiltrat", r"crazy paving", r"tree", r"\bbud", r"dependent",
        r"consolidat", r"reticulonodular"],
    "Pulmonary fibrotic sequela": [r"fibro", r"sequela", r"sequelae", r"\bscar", r"traction",
        r"honeycomb", r"reticul", r"pleuroparenchymal", r"retraction", r"recession",
        r"structural distortion", r"architectural distortion", r"volume loss"],
    "Pleural effusion": [r"pleural effusion", r"pleural fluid", r"effusion", r"hydrothorax",
        r"pleural space", r"\bpleura"],
    "Mosaic attenuation pattern": [r"mosaic"],
    "Peribronchial thickening": [r"peribronch", r"peribroch", r"bronchial wall",
        r"wall thickness", r"thickness increase", r"cuffing", r"bronchial thickening",
        r"bronchovascular"],
    "Consolidation": [r"consolidat", r"infiltrat", r"air bronchogram", r"pneumoni",
        r"bronchopneumon"],
    "Bronchiectasis": [r"bronchiect", r"bronchial (structures? )?(dilat|enlarg|ectas)",
        r"dilat.{0,15}bronch", r"bronch.{0,15}dilat", r"enlarg.{0,25}bronchial", r"ectasia",
        r"ectatic"],
    "Interlobular septal thickening": [r"septal", r"\bsepta\b", r"kerley", r"interlobular",
        r"crazy paving", r"reticul"],
}

# 否定の手がかり（別スタイルで表示）
NEGATION = [r"\bno\b", r"\bnot\b", r"\bnone\b", r"without", r"absent", r"\bfree of\b",
            r"did not", r"was not", r"were not", r"\bnor\b", r"unremarkable",
            r"not (detected|observed|seen|identified|present)", r"(do|does|did) not reach",
            r"not reach", r"below 1 ?cm", r"not exceed", r"negative"]


# ── 日本語訳ペイン用キーワード ───────────────────────────────────────────
# 訳文中の実際の表記を数えて作成（例: 心囊液 は「囊」の字体、心拡大は「心臓の輪郭」とも訳される）
KEYWORDS_JA: dict[str, list[str]] = {
    "Medical material": ["人工物", "プロテーゼ", "プロテーシス", "バイパス", "手術", "ペースメーカー", "カテーテル", "ポート", "ステント", "気管内チューブ",
        "経鼻胃管", "人工弁", "クリップ", "ステープル", "縫合", "金属", "異物", "インプラント",
        "カニューレ", "胸骨正中切開", "気管切開", "固定材", "セルクラージュ", "弁形成", "スクリュー",
        "メッシュ", "グラフト", "コイル", "ドレーン", "電極", "材料", "高吸収", "術後", "留置"],
    "Arterial wall calcification": ["石灰", "アテローム", "動脈硬化", "大動脈", "腕頭", "鎖骨下"],
    "Cardiomegaly": ["心胸郭指数", "心胸郭", "心陰影", "心拡大", "心臓", "心腔", "心房", "心室", "心胸郭比", "心胸比", "CTO"],
    "Pericardial effusion": ["心囊", "心嚢", "心膜"],
    "Coronary artery wall calcification": ["石灰", "冠動脈", "LAD", "RCA", "左前下行", "右冠動脈", "回旋枝"],
    "Hiatal hernia": ["食道裂孔", "ヘルニア", "裂孔"],
    "Lymphadenopathy": ["リンパ節", "LAP", "傍気管", "前血管", "気管分岐下", "大動脈肺動脈",
        "肺門", "腋窩", "鎖骨上", "ミリメートル", "病的", "短軸", "短径", "脂肪性", "リンパ門"],
    "Emphysema": ["肺気腫", "気腫", "ブラ", "ブレブ", "気道捕捉", "エアトラッピング", "中心細葉",
        "傍中隔", "汎細葉", "気囊", "含気嚢"],
    "Atelectasis": ["無気肺", "含気低下", "含気", "虚脱", "容積減少", "線維無気肺", "無気肺性"],
    "Lung nodule": ["結節", "腫瘤", "転移", "微小結節", "スピキュラ", "病変", "GGN"],
    "Lung opacity": ["すりガラス", "濃度上昇", "濃度増加", "濃度", "陰影", "浸潤", "crazy paving",
        "敷石", "木の芽", "tree-in-bud", "背側", "網状", "スムース"],
    "Pulmonary fibrotic sequela": ["陳旧", "胸膜実質", "線維", "瘢痕", "後遺", "牽引", "蜂巣", "網状", "胸膜肺",
        "収縮", "構築", "容積減少", "変形"],
    "Pleural effusion": ["胸水", "胸膜液", "胸腔", "胸膜"],
    "Mosaic attenuation pattern": ["モザイク"],
    "Peribronchial thickening": ["気管支壁", "気管支周囲", "壁肥厚", "肥厚", "気管支血管",
        "厚み", "壁厚"],
    "Consolidation": ["腺房", "浸潤影", "浸潤", "air bronchogram", "肺炎", "気管支肺炎", "コンソリデーション",
        "エアブロンコグラム"],
    "Bronchiectasis": ["気管支拡張", "気管支.{0,6}拡張", "拡張", "気管支.{0,6}ectas"],
    "Interlobular septal thickening": ["クレイジー", "ペイビング", "小葉間隔壁", "隔壁", "Kerley", "カーリー", "網状",
        "crazy paving", "敷石"],
}

NEGATION_JA = ["認められ(なかった|ない|ず)", "みられ(なかった|ない|ず)", "見られ(なかった|ない|ず)",
    "指摘され(なかった|ない|ず)", "検出され(なかった|ない|ず)", "同定され(なかった|ない|ず)",
    "確認され(なかった|ない|ず)", "存在しない", "なかった", "陰性", "正常", "自然",
    "達しない", "達していない", "超えない", "以下", "未満", "なし", "伴わない", "欠如", "除外"]


VALUES = [
    ("present", "有", "present", "現在の検査で存在・観察・検出と記載"),
    ("uncertain", "疑", "uncertain", "possible / suspected / cannot exclude / ? などのヘッジ表現"),
    ("negated", "否", "negated", "明示的に否定（no …, not detected, 病的サイズに達しない 等）"),
    ("not_mentioned", "無", "not mentioned", "どちらのセクションにも言及がない"),
]

HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>CT-RATE アノテーション（__READER__）</title>
<style>
:root{
  --present:#0072B2; --uncertain:#E69F00; --negated:#7B4FA3; --notment:#9AA0A6;
  --ink:#1b1b1b; --line:#d9dce1; --bg:#f7f8fa; --panel:#ffffff; --accent:#C55A11;
}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Noto Sans JP",sans-serif;
     color:var(--ink);background:var(--bg);font-size:14px}
header{position:sticky;top:0;z-index:10;background:var(--panel);border-bottom:2px solid var(--line);
       padding:8px 14px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
header .grow{flex:1}
h1{font-size:15px;margin:0;font-weight:700}
button{font:inherit;padding:5px 11px;border:1px solid var(--line);background:#fff;border-radius:6px;cursor:pointer}
button:hover{background:#eef1f5}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:700}
input[type=text]{font:inherit;padding:4px 7px;border:1px solid var(--line);border-radius:5px}
#wrap{display:grid;grid-template-columns:minmax(380px,44%) 1fr;gap:12px;padding:12px;align-items:start}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
#report{position:sticky;top:64px;max-height:calc(100vh - 88px);overflow:auto}
.sec{font-weight:700;color:var(--accent);margin:12px 0 4px;font-size:12px;letter-spacing:.04em}
.body{white-space:pre-wrap;line-height:1.75;user-select:text}
.ja{background:#fffdf5;border-left:3px solid var(--accent);padding:8px 10px;margin-top:6px;
    line-height:1.8;white-space:pre-wrap;color:#333}
table{width:100%;border-collapse:collapse}
th,td{border-bottom:1px solid var(--line);padding:5px 4px;vertical-align:top}
th{text-align:left;font-size:11px;color:#555;font-weight:600}
tr.row.active{background:#eef4fb;outline:2px solid var(--present);outline-offset:-2px}
tr.row.unset td:first-child{border-left:4px solid var(--accent)}
.cat{font-weight:600;line-height:1.35}
.cat .en{font-size:11px;color:#666;font-weight:400;display:block}
.vals{display:flex;gap:4px}
.vals label{border:1px solid var(--line);border-radius:6px;padding:3px 8px;cursor:pointer;
            font-weight:700;background:#fff;min-width:34px;text-align:center;user-select:none}
.vals input{display:none}
.vals label.on[data-v=present]{background:var(--present);color:#fff;border-color:var(--present)}
.vals label.on[data-v=uncertain]{background:var(--uncertain);color:#222;border-color:var(--uncertain)}
.vals label.on[data-v=negated]{background:var(--negated);color:#fff;border-color:var(--negated)}
.vals label.on[data-v=not_mentioned]{background:var(--notment);color:#fff;border-color:var(--notment)}
.ev{width:100%;margin-top:3px}
.flag{font-size:11px;color:#444;margin-top:3px;display:flex;gap:5px;align-items:center}
.help{cursor:help;border-bottom:1px dotted #999}
mark.kw{background:#FFF2A8;color:#111;border-radius:2px;padding:0 1px}
mark.neg{background:transparent;color:#0072B2;font-weight:700;
         border-bottom:2px dotted #0072B2;padding-bottom:1px}
.hits{display:inline-block;min-width:18px;text-align:center;font-size:10px;font-weight:700;
      color:#7a3b08;background:#FFF2A8;border-radius:9px;padding:1px 5px;margin-left:5px}
.hint{font-size:11px;color:#666;line-height:1.6}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;background:#eef1f5;font-size:11px}
#pager{display:flex;gap:8px;align-items:center;justify-content:space-between;margin-top:10px}
dialog{border:1px solid var(--line);border-radius:10px;max-width:620px;padding:16px}
dialog h3{margin:0 0 6px;font-size:14px}
</style>
</head>
<body>
<header>
  <h1>CT-RATE confirmatory set <span style="font-weight:400;color:#666">/ __READER__</span></h1>
  <span>読影者 <input type="text" id="annotator" value="__READER__" size="12"></span>
  <span class="badge" id="progress">0 / 0</span>
  <span class="badge" id="unsetBadge">未入力 0</span>
  <span class="grow"></span>
  <label class="hint"><input type="checkbox" id="showJa" checked> 日本語訳</label>
  <label class="hint"><input type="checkbox" id="marker"> マーカー</label>
  <label class="hint"><input type="checkbox" id="autoNM"> 開いたとき全て「無」で初期化</label>
  <button id="btnGuide">定義</button>
  <button id="btnJson">作業状態を保存</button>
  <button id="btnLoad">読み込み</button>
  <button class="primary" id="btnCsv">CSV 書き出し</button>
</header>

<div id="lsWarn" style="display:none;background:#fff4e5;border-bottom:2px solid #C55A11;
     padding:8px 14px;color:#7a3b08;font-weight:700">
  このブラウザでは自動保存（localStorage）が使えません。作業が失われないよう、
  <b>こまめに「作業状態を保存」で JSON を書き出してください</b>。
  Chrome / Firefox / Edge で開くと自動保存が有効になります。
</div>

<div id="wrap">
  <div class="panel" id="report">
    <div><span class="badge" id="rid"></span></div>
    <div class="sec">FINDINGS</div><div class="body" id="findings"></div>
    <div class="jaBox" id="jaFindingsBox"><div class="ja" id="jaFindings"></div></div>
    <div class="sec">IMPRESSION</div><div class="body" id="impression"></div>
    <div class="jaBox" id="jaImpressionBox"><div class="ja" id="jaImpression"></div></div>
  </div>

  <div class="panel">
    <div class="hint" style="margin-bottom:8px">
      キーボード: <b>1</b> 有 / <b>2</b> 疑 / <b>3</b> 否 / <b>4</b> 無 &nbsp;·&nbsp;
      <b>↑ ↓</b> 行移動 &nbsp;·&nbsp; <b>Shift + ← →</b> レポート移動 &nbsp;·&nbsp;
      <b>E</b> 選択したレポート本文を根拠欄に貼付
    </div>
    <table>
      <thead><tr><th style="width:38%">カテゴリ</th><th>判定・根拠</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
    <div id="pager">
      <button id="prev">← 前</button>
      <button id="fillNM">残りを「無」にする</button>
      <input type="text" id="jump" size="4" placeholder="#">
      <button id="next" class="primary">次 →</button>
    </div>
  </div>
</div>

<dialog id="guide"><h3>ラベル定義</h3><div id="guideBody" class="hint"></div>
<div style="text-align:right;margin-top:10px"><button onclick="guide.close()">閉じる</button></div></dialog>

<script id="REPORTS" type="application/json">__REPORTS__</script>
<script id="LABELS" type="application/json">__LABELS__</script>
<script id="VALUES" type="application/json">__VALUES__</script>
<script id="KEYWORDS" type="application/json">__KEYWORDS__</script>
<script id="NEGATION" type="application/json">__NEGATION__</script>
<script id="KEYWORDS_JA" type="application/json">__KEYWORDS_JA__</script>
<script id="NEGATION_JA" type="application/json">__NEGATION_JA__</script>
<script>
const REPORTS = JSON.parse(document.getElementById('REPORTS').textContent);
const LABELS  = JSON.parse(document.getElementById('LABELS').textContent);
const VALUES  = JSON.parse(document.getElementById('VALUES').textContent);
const KEYWORDS= JSON.parse(document.getElementById('KEYWORDS').textContent);
const NEGATION= JSON.parse(document.getElementById('NEGATION').textContent);
const KEYWORDS_JA = JSON.parse(document.getElementById('KEYWORDS_JA').textContent);
const NEGATION_JA = JSON.parse(document.getElementById('NEGATION_JA').textContent);
const RX = {}; Object.keys(KEYWORDS).forEach(k=>{ RX[k]=new RegExp('('+KEYWORDS[k].join('|')+')','gi'); });
const RXJA = {}; Object.keys(KEYWORDS_JA).forEach(k=>{ RXJA[k]=new RegExp('('+KEYWORDS_JA[k].join('|')+')','gi'); });
const RXNEG = new RegExp('('+NEGATION.join('|')+')','gi');
const RXNEGJA = new RegExp('('+NEGATION_JA.join('|')+')','gi');
const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

function ranges(text, re){
  const out=[]; re.lastIndex=0; let m;
  while((m=re.exec(text))!==null){ if(m[0].length===0){re.lastIndex++;continue;}
    out.push([m.index, m.index+m[0].length]); }
  return out;
}
function merge(rs){
  rs.sort((a,b)=>a[0]-b[0]); const o=[];
  rs.forEach(r=>{ const l=o[o.length-1];
    if(l && r[0]<=l[1]){ l[1]=Math.max(l[1],r[1]); } else o.push([r[0],r[1]]); });
  return o;
}
function highlight(text, cat, lang){
  if(!text) return '';
  const dict = (lang==='ja') ? RXJA : RX, neg = (lang==='ja') ? RXNEGJA : RXNEG;
  const kw = merge(ranges(text, dict[cat]||/$^/g)).map(r=>[r[0],r[1],'kw']);
  const ng = merge(ranges(text, neg)).map(r=>[r[0],r[1],'neg']);
  // キーワードと否定が重なる場合はキーワードを優先
  const keep = ng.filter(n=>!kw.some(k=>n[0]<k[1]&&k[0]<n[1]));
  const all = kw.concat(keep).sort((a,b)=>a[0]-b[0]);
  let out='', pos=0;
  all.forEach(([a,b,cls])=>{ if(a<pos) return;
    out += esc(text.slice(pos,a)) + '<mark class="'+cls+'">' + esc(text.slice(a,b)) + '</mark>'; pos=b; });
  return out + esc(text.slice(pos));
}
function countHits(text, cat){ return merge(ranges(text, RX[cat]||/$^/g)).length; }
const READER = "__READER__";
const KEY = 'ctrate_confirmatory_annotation_v1__' + (READER || 'anon');
let LS_OK = true;
try { localStorage.setItem(KEY+'__probe','1'); localStorage.removeItem(KEY+'__probe'); }
catch(err){ LS_OK = false; }
let st = {};
try { st = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch(err){ st = {}; }
if(!st.data) st = {annotator:READER, data:{}, idx:0};
if(!st.annotator) st.annotator = READER;
let idx = st.idx || 0, activeRow = 0;

const $ = id => document.getElementById(id);
let dirty = 0;
const save = () => {
  st.idx = idx;
  if(LS_OK){ try { localStorage.setItem(KEY, JSON.stringify(st)); } catch(err){ LS_OK=false; warnLS(); } }
  else { dirty++; if(dirty % 50 === 0) warnLS(); }
};
function warnLS(){
  const b = $('lsWarn');
  if(b) b.style.display = '';
}

function recOf(vol){ if(!st.data[vol]) st.data[vol] = {}; return st.data[vol]; }

function buildRows(){
  const tb = $('rows'); tb.innerHTML='';
  LABELS.forEach((L,i)=>{
    const tr = document.createElement('tr');
    tr.className='row'; tr.dataset.i=i;
    tr.innerHTML = `<td><div class="cat"><span class="help" title="${L.def.replace(/"/g,'&quot;')}">${L.ja}</span>
        <span class="en">${L.en}</span></div><span class="hits" style="display:none"></span></td>
      <td>
        <div class="vals" data-i="${i}">
          ${VALUES.map(v=>`<label data-v="${v.key}" title="${v.hint}"><input type="radio" name="v${i}" value="${v.key}">${v.jp}</label>`).join('')}
        </div>
        <input class="ev" type="text" data-i="${i}" placeholder="根拠（レポートからの引用）">
        <div class="flag" data-i="${i}"><label><input type="checkbox" class="tp"> 前回検査のみ／今回消失</label></div>
      </td>`;
    tr.addEventListener('mousedown',()=>{activeRow=i; paintActive();});
    tb.appendChild(tr);
  });
  tb.querySelectorAll('.vals label').forEach(lb=>{
    lb.addEventListener('click',e=>{
      const i = +lb.parentNode.dataset.i; setVal(i, lb.dataset.v); activeRow=i; paintActive();
    });
  });
  tb.querySelectorAll('input.ev').forEach(inp=>{
    inp.addEventListener('input',()=>{ const i=+inp.dataset.i;
      recOf(cur().vol)[LABELS[i].en] = Object.assign(recOf(cur().vol)[LABELS[i].en]||{}, {evidence: inp.value}); save(); });
  });
  tb.querySelectorAll('input.tp').forEach(cb=>{
    cb.addEventListener('change',()=>{ const i=+cb.closest('.flag').dataset.i;
      recOf(cur().vol)[LABELS[i].en] = Object.assign(recOf(cur().vol)[LABELS[i].en]||{},
        {temporality: cb.checked ? 'prior_or_resolved' : 'current'}); save(); });
  });
}

const cur = () => REPORTS[idx];

function setVal(i, v){
  const r = recOf(cur().vol), k = LABELS[i].en;
  r[k] = Object.assign(r[k]||{}, {value:v});
  if(v==='not_mentioned'){ r[k].evidence=''; r[k].temporality=''; }
  if(v==='negated'){ r[k].temporality=''; }
  if((v==='present'||v==='uncertain') && !r[k].temporality) r[k].temporality='current';
  save(); paintRows();
}

function paintActive(){
  document.querySelectorAll('tr.row').forEach(tr=>tr.classList.toggle('active', +tr.dataset.i===activeRow));
  renderBody();
}

function paintHits(){
  const R = cur(), on = $('marker').checked, t = (R.findings||'') + ' ' + (R.impression||'');
  LABELS.forEach((L,i)=>{
    const el = document.querySelector(`tr.row[data-i="${i}"] .hits`);
    if(!el) return;
    if(!on){ el.style.display='none'; return; }
    const n = countHits(t, L.en);
    el.style.display=''; el.textContent = n;
    el.style.visibility = n ? 'visible' : 'hidden';
  });
}

function paintRows(){
  const r = recOf(cur().vol);
  let unset=0;
  LABELS.forEach((L,i)=>{
    const rec = r[L.en]||{}, tr = document.querySelector(`tr.row[data-i="${i}"]`);
    tr.querySelectorAll('.vals label').forEach(lb=>lb.classList.toggle('on', lb.dataset.v===rec.value));
    const ev = tr.querySelector('input.ev'), fl = tr.querySelector('.flag'), cb = tr.querySelector('input.tp');
    ev.value = rec.evidence || '';
    ev.style.display = (rec.value && rec.value!=='not_mentioned') ? '' : 'none';
    fl.style.display = (rec.value==='present'||rec.value==='uncertain') ? '' : 'none';
    cb.checked = rec.temporality === 'prior_or_resolved';
    if(!rec.value) unset++;
    tr.classList.toggle('unset', !rec.value);
  });
  $('unsetBadge').textContent = '未入力 ' + unset;
  paintHits();
  const done = REPORTS.filter(x=>{
    const rr = st.data[x.vol]||{}; return LABELS.every(L=>rr[L.en] && rr[L.en].value);
  }).length;
  $('progress').textContent = `${done} / ${REPORTS.length} 完了`;
  paintActive();
}

function render(){
  const R = cur();
  $('rid').textContent = `#${R.order}  (${idx+1}/${REPORTS.length})`;
  renderBody();
  toggleJa();
  if($('autoNM').checked){
    const r = recOf(R.vol);
    LABELS.forEach(L=>{ if(!r[L.en] || !r[L.en].value) r[L.en]={value:'not_mentioned',evidence:'',temporality:''}; });
    save();
  }
  activeRow = 0; paintRows(); $('report').scrollTop = 0; $('jump').value = idx+1;
}

function renderBody(){
  const R = cur(), on = $('marker').checked, cat = LABELS[activeRow] ? LABELS[activeRow].en : null;
  if(on && cat){
    $('findings').innerHTML   = highlight(R.findings || '(なし)', cat, 'en');
    $('impression').innerHTML = highlight(R.impression || '(なし)', cat, 'en');
    $('jaFindings').innerHTML   = highlight(R.ja_findings || '（日本語訳なし）', cat, 'ja');
    $('jaImpression').innerHTML = highlight(R.ja_impression || '（日本語訳なし）', cat, 'ja');
  } else {
    $('findings').textContent   = R.findings || '(なし)';
    $('impression').textContent = R.impression || '(なし)';
    $('jaFindings').textContent   = R.ja_findings || '（日本語訳なし）';
    $('jaImpression').textContent = R.ja_impression || '（日本語訳なし）';
  }
}

function toggleJa(){
  const on = $('showJa').checked;
  document.querySelectorAll('.jaBox').forEach(e=>e.style.display = on ? '' : 'none');
}

function go(d){ idx = Math.min(REPORTS.length-1, Math.max(0, idx+d)); save(); render(); }

$('prev').onclick=()=>go(-1); $('next').onclick=()=>go(1);
$('jump').addEventListener('change',()=>{ const n=parseInt($('jump').value,10);
  if(n>=1&&n<=REPORTS.length){ idx=n-1; save(); render(); } });
$('showJa').onchange=toggleJa;
$('marker').onchange=()=>{ renderBody(); paintHits(); };
$('fillNM').onclick=()=>{ const r=recOf(cur().vol);
  LABELS.forEach(L=>{ if(!r[L.en]||!r[L.en].value) r[L.en]={value:'not_mentioned',evidence:'',temporality:''}; });
  save(); paintRows(); };
$('annotator').value = st.annotator||READER;
$('annotator').addEventListener('input',()=>{ st.annotator=$('annotator').value; save(); });

document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT' && e.target.type==='text') return;
  if(e.shiftKey && e.key==='ArrowRight'){ go(1); e.preventDefault(); return; }
  if(e.shiftKey && e.key==='ArrowLeft'){ go(-1); e.preventDefault(); return; }
  if(e.key==='ArrowDown'){ activeRow=Math.min(LABELS.length-1,activeRow+1); paintActive(); e.preventDefault(); return; }
  if(e.key==='ArrowUp'){ activeRow=Math.max(0,activeRow-1); paintActive(); e.preventDefault(); return; }
  const m = {'1':'present','2':'uncertain','3':'negated','4':'not_mentioned'};
  if(m[e.key]){ setVal(activeRow, m[e.key]);
    if(e.key!=='0'){ activeRow=Math.min(LABELS.length-1,activeRow+1); paintActive(); } e.preventDefault(); return; }
  if(e.key==='e'||e.key==='E'){
    const sel=String(window.getSelection()).trim();
    if(sel){ const inp=document.querySelector(`input.ev[data-i="${activeRow}"]`);
      inp.value = sel; inp.dispatchEvent(new Event('input')); }
    e.preventDefault();
  }
});

$('btnCsv').onclick=()=>{
  const q = s => '"' + String(s==null?'':s).replace(/"/g,'""') + '"';
  const now = new Date().toISOString().replace('T',' ').slice(0,19);
  const out = ['volume_name,annotator,label_name,value,temporality,evidence,updated_at'];
  REPORTS.forEach(R=>{
    const r = st.data[R.vol]||{};
    LABELS.forEach(L=>{
      const rec = r[L.en]; if(!rec || !rec.value) return;
      const tp = (rec.value==='present'||rec.value==='uncertain') ? (rec.temporality||'current') : '';
      out.push([q(R.vol),q(st.annotator||''),q(L.en),q(rec.value),q(tp),q(rec.evidence||''),q(now)].join(','));
    });
  });
  dl(out.join('\n'), `confirmatory250_annotation_${(st.annotator||'anon')}.csv`, 'text/csv');
};
$('btnJson').onclick=()=>dl(JSON.stringify(st), `annotation_state_${(st.annotator||'anon')}.json`, 'application/json');
$('btnLoad').onclick=()=>{
  const f=document.createElement('input'); f.type='file'; f.accept='.json';
  f.onchange=()=>{ const fr=new FileReader();
    fr.onload=()=>{ try{ st=JSON.parse(fr.result); idx=st.idx||0; save();
      $('annotator').value=st.annotator||''; render(); }catch(err){ alert('読み込み失敗: '+err); } };
    fr.readAsText(f.files[0]); };
  f.click();
};
function dl(text, name, mime){
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([text],{type:mime+';charset=utf-8'}));
  a.download=name; a.click(); URL.revokeObjectURL(a.href);
}
$('btnGuide').onclick=()=>{
  $('guideBody').innerHTML = '<p>' + VALUES.map(v=>`<b>${v.jp}</b> = ${v.en}（${v.hint}）`).join('<br>') + '</p>'
    + '<p>所見が前回検査にのみ存在する、または今回消失・退縮している場合は「前回検査のみ／今回消失」にチェック。'
    + 'stable / unchanged は現在も存在するのでチェック不要。</p>'
    + '<p><b>マーカー</b>を on にすると、現在選択している行のカテゴリに関係しそうな語が'
    + '<mark class="kw">黄色</mark>で、否定の手がかりが<mark class="neg">青の点線</mark>で強調されます。'
    + '行の右の数字は該当語の件数です。単純なキーワード一致なので、無関係な語が光ることも、'
    + '光らない記載もあります。最終判断は本文を読んで行ってください。</p><hr>'
    + LABELS.map(L=>`<p><b>${L.ja}</b>（${L.en}）<br>${L.def}</p>`).join('');
  guide.showModal();
};

if(!LS_OK) warnLS();
window.addEventListener('beforeunload', e=>{
  if(!LS_OK){ e.preventDefault(); e.returnValue=''; }
});
buildRows(); render();
</script>
</body>
</html>
"""


READERS = ["yamagishi", "takamatsu"]


def build_handout() -> str:
    """読影者に渡す定義ハンドアウト（CT-RATE / LLM ラベルの情報を含まない）。"""
    L = ["# ラベル定義（confirmatory reference set）", "",
         "判断はレポート本文（FINDINGS と IMPRESSION）のみに基づいてください。",
         "臨床背景・疫学・一般的な医学知識からの推測はしないでください。", "",
         "## 4値の区別", ""]
    for k, jp, en, h in VALUES:
        L.append(f"- **{jp}**（{en}）… {h}")
    L += ["", "## 共通ルール", "",
          "1. 明示的な否定（no …, not detected, absent など）は「否」、言及がなければ「無」。",
          "2. ヘッジ表現（possible / suspected / cannot exclude / ? など）は「疑」。",
          "3. 所見が前回検査にのみ存在する、または今回消失・退縮している場合は「前回検査のみ／今回消失」にチェック。"
          "stable / unchanged は現在も存在するのでチェック不要。",
          "4. 撮像範囲内（下部頸部・腋窩・上腹部を含む）の所見は対象。ただし肺実質を要件とするカテゴリは肺実質の所見に限る。",
          "5. 1つの記述が複数カテゴリを満たすことがある。該当するものはすべて「有」にする。",
          "6. ある語が他の所見の分布を述べているだけの場合、その語のカテゴリは立てない。", "",
          "## カテゴリ別定義", ""]
    for en, jp, d in LABELS:
        L += [f"### {jp}（{en}）", "", d, ""]
    return "\n".join(L)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    design = pd.read_csv(DATA / "confirmatory250_design.csv")
    rep = pd.read_csv(DATA / "CT-RATE_reports.csv",
                      usecols=["VolumeName", "Findings_EN", "Impressions_EN"])
    m = design.merge(rep, on="VolumeName", how="left").sort_values("reading_order")

    ja_path = DATA / "confirmatory250_reports_ja.json"
    ja = json.loads(ja_path.read_text(encoding="utf-8")) if ja_path.exists() else {}

    reports = []
    for _, r in m.iterrows():
        j = ja.get(r.VolumeName, {})
        reports.append({
            "vol": r.VolumeName,
            "order": int(r.reading_order),
            "findings": str(r.Findings_EN or "").strip(),
            "impression": str(r.Impressions_EN or "").strip(),
            "ja_findings": j.get("findings", ""),
            "ja_impression": j.get("impression", ""),
        })

    labels = [{"en": en, "ja": jp, "def": d} for en, jp, d in LABELS]
    values = [{"key": k, "jp": jp, "en": en, "hint": h} for k, jp, en, h in VALUES]

    def emb(obj):
        return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")

    base = (HTML.replace("__REPORTS__", emb(reports))
                .replace("__LABELS__", emb(labels))
                .replace("__VALUES__", emb(values))
                .replace("__KEYWORDS__", emb(KEYWORDS))
                .replace("__NEGATION__", emb(NEGATION))
                .replace("__KEYWORDS_JA__", emb(KEYWORDS_JA))
                .replace("__NEGATION_JA__", emb(NEGATION_JA)))

    n_ja = sum(1 for x in reports if x["ja_findings"])
    for reader in READERS:
        p = OUT / f"annotate_{reader}.html"
        p.write_text(base.replace("__READER__", reader), encoding="utf-8")
        print(f"→ {p.name}  ({p.stat().st_size/1024:.0f} KB)  reader={reader}")

    h = OUT / "label_definitions_ja.md"
    h.write_text(build_handout(), encoding="utf-8")
    print(f"→ {h.name}")
    print(f"   レポート {len(reports)} 件 / カテゴリ {len(labels)} / 日本語訳あり {n_ja} 件")


if __name__ == "__main__":
    main()
