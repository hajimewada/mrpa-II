# MRPA-II Scripts Guide

この文書は、`scripts/` 配下に追加された次の3スクリプトの目的、前提条件、使用法、入出力をまとめたものです。

```text
scripts/
├── process_jsut.sh
├── visualize_mrpa_utterance.py
└── generate_mrpa_index.py
```

## 全体ワークフロー

```text
JSUT WAV
    ↓ process_jsut.sh
観測F0・time・metadata
    ↓
補間F0・voicing mask・interpolation mask
    ↓
CWT
    ↓
ScaleProfile
    ↓
Boundary candidates
    ↓
Nucleus candidates
    ↓ visualize_mrpa_utterance.py
Phone・本文・F0・ScaleProfile・Boundary・Nucleus可視化
    ↓ generate_mrpa_index.py
検索可能な発話一覧HTML
```

---

## 1. `process_jsut.sh`

### 目的

JSUTのWAVファイルを発話単位で処理し、MRPA-IIの完全な解析パイプラインを実行します。

```text
WAV
→ observed F0
→ interpolated continuous F0
→ CWT
→ ScaleProfile
→ Boundary candidates
→ Nucleus candidates
```

各発話は独立したディレクトリへ保存されます。全成果物の生成に成功した場合だけ、発話ディレクトリ内に`.complete` markerを作成します。

### 既定のディレクトリ

```text
JSUT data root:
~/projects/mrpa/data/jsut

JSUT corpus root:
~/projects/mrpa/data/jsut/raw/jsut_ver1.1

MRPA-II output root:
~/projects/mrpa/data/jsut/results/mrpa-II
```

### 前提条件

- MRPA-IIのPython仮想環境が`.venv/`に存在すること
- JSUT v1.1が配置されていること
- 次の解析moduleが存在すること

```text
analysis.f0.extract_real_f0
analysis.f0.interpolate_f0
analysis.f0.extract_cwt
analysis.f0.extract_scale_profile
analysis.boundary.extract_boundary_candidates
analysis.nucleus.extract_nucleus_candidates
```

### 基本実行

最初の1発話だけを処理します。

```bash
SUBSET=basic5000 \
LIMIT=1 \
FORCE=1 \
scripts/process_jsut.sh
```

10発話を処理します。

```bash
SUBSET=basic5000 \
LIMIT=10 \
FORCE=1 \
scripts/process_jsut.sh
```

subset全体を処理します。

```bash
SUBSET=basic5000 \
LIMIT=0 \
FORCE=0 \
scripts/process_jsut.sh
```

### 環境変数

#### `JSUT_DATA_ROOT`

JSUT関連データの管理rootです。

```text
Default: ~/projects/mrpa/data/jsut
```

#### `JSUT_CORPUS_ROOT`

JSUT v1.1のコーパスrootです。

```text
Default: $JSUT_DATA_ROOT/raw/jsut_ver1.1
```

#### `OUTPUT_ROOT`

MRPA-II成果物の保存先です。

```text
Default: $JSUT_DATA_ROOT/results/mrpa-II
```

#### `SUBSET`

処理対象のJSUT subsetです。

```text
Default: basic5000
```

#### `LIMIT`

走査する最大発話数です。

```text
0: 全発話
1: 最初の1発話
10: 最初の10発話
```

#### `FORCE`

完了済み発話の再処理を制御します。

```text
0: .completeを持つ発話をskip
1: 完了済みでも再処理
```

#### `PYTHON`

使用するPython interpreterです。

```text
Default: <repository-root>/.venv/bin/python
```

### 発話ごとの出力構造

```text
results/mrpa-II/basic5000/BASIC5000_0001/
├── .complete
├── f0/
│   ├── BASIC5000_0001_f0.npy
│   ├── BASIC5000_0001_f0_time.npy
│   ├── BASIC5000_0001_f0_metadata.json
│   ├── BASIC5000_0001_f0_interpolated.npy
│   ├── BASIC5000_0001_voicing.npy
│   ├── BASIC5000_0001_interpolation_mask.npy
│   └── BASIC5000_0001_interpolation_metadata.json
├── cwt/
│   ├── BASIC5000_0001_cwt.npz
│   └── BASIC5000_0001_cwt_metadata.json
├── scale_profile/
│   ├── BASIC5000_0001_scale_profile.npz
│   └── BASIC5000_0001_scale_profile_metadata.json
├── boundary/
│   ├── BASIC5000_0001_boundary_candidates.npz
│   └── BASIC5000_0001_boundary_candidates_metadata.json
├── nucleus/
│   ├── BASIC5000_0001_nucleus_candidates.npz
│   └── BASIC5000_0001_nucleus_candidates_metadata.json
└── logs/
    └── pipeline.log
```

### ログ確認

```bash
cat \
  "$HOME/projects/mrpa/data/jsut/results/mrpa-II/basic5000/BASIC5000_0001/logs/pipeline.log"
```

正常終了したログには次が記録されます。

```text
pipeline_status=complete
```

### バッチ成果物数の確認

```bash
RESULT_ROOT="$HOME/projects/mrpa/data/jsut/results/mrpa-II/basic5000"

printf 'complete=%s\n' \
  "$(find "$RESULT_ROOT" -name '.complete' -type f | wc -l)"

printf 'cwt_npz=%s\n' \
  "$(find "$RESULT_ROOT" -path '*/cwt/*_cwt.npz' -type f | wc -l)"

printf 'scale_profile_npz=%s\n' \
  "$(find "$RESULT_ROOT" -path '*/scale_profile/*_scale_profile.npz' -type f | wc -l)"

printf 'boundary_npz=%s\n' \
  "$(find "$RESULT_ROOT" -path '*/boundary/*_boundary_candidates.npz' -type f | wc -l)"

printf 'nucleus_npz=%s\n' \
  "$(find "$RESULT_ROOT" -path '*/nucleus/*_nucleus_candidates.npz' -type f | wc -l)"
```

### 検証済み結果

```text
basic5000 10発話
processed=10
failed=0

complete markers       10
CWT archives           10
ScaleProfile archives  10
Boundary archives      10
Nucleus archives       10
complete logs          10
```

---

## 2. `visualize_mrpa_utterance.py`

### 目的

1発話のMRPA-II成果物を共通時間軸で可視化し、PNGと自己完結型HTMLを生成します。

生成される図には次が含まれます。

```text
Phone forced alignment
本文token alignment
Observed / interpolated F0
ScaleProfile magnitude
Boundary strength and candidates
Nucleus strength and candidates
```

### alignmentの考え方

本文を時間へ均等配置するのではなく、次の経路で整列します。

```text
JSUT本文
→ OpenJTalk形態素解析・G2P
→ 本文tokenのPhone列
→ jsut-labのforced-alignment Phone列
→ global sequence alignment
→ tokenごとの開始・終了時刻
```

### 追加の前提条件

#### `pyopenjtalk`

```bash
.venv/bin/python -m pip install pyopenjtalk
```

確認:

```bash
.venv/bin/python - <<'PY'
import pyopenjtalk

print(pyopenjtalk.g2p("こんにちは"))
PY
```

#### JSUT forced-alignment labels

`external/jsut-lab`はローカル参照用です。MRPA-II repositoryへコミットしません。

```bash
mkdir -p external

git clone \
  https://github.com/r9y9/jsut-lab.git \
  external/jsut-lab
```

`.gitignore`には次を含めます。

```text
/external/jsut-lab/
```

#### 日本語フォント

Ubuntu/DebianでNoto CJKを使用する例:

```bash
sudo apt install -y fonts-noto-cjk
fc-cache -f
```

フォント確認:

```bash
fc-match \
  --format='%{family}: %{file}\n' \
  ':charset=65e5'
```

### BASIC5000_0001の可視化

```bash
cd "$HOME/projects/mrpa-II"

UTTERANCE_DIR="$HOME/projects/mrpa/data/jsut/results/mrpa-II/basic5000/BASIC5000_0001"
LAB_FILE="$HOME/projects/mrpa-II/external/jsut-lab/basic5000/lab/BASIC5000_0001.lab"
FONT_FILE="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

.venv/bin/python \
  scripts/visualize_mrpa_utterance.py \
  --input-dir "$UTTERANCE_DIR" \
  --corpus-root \
  "$HOME/projects/mrpa/data/jsut/raw/jsut_ver1.1" \
  --lab-file "$LAB_FILE" \
  --font-file "$FONT_FILE" \
  --output "$UTTERANCE_DIR/BASIC5000_0001_aligned.png" \
  --html-output "$UTTERANCE_DIR/BASIC5000_0001_aligned.html" \
  --label-stride 5
```

### 主なoption

#### `--input-dir`

`process_jsut.sh`が作成した発話ディレクトリです。

#### `--corpus-root`

JSUT v1.1 rootです。本文ファイルの自動検出に使います。

#### `--transcript-file`

既定位置以外の`transcript_utf8.txt`を使う場合に指定します。

#### `--lab-file`

HTK/HTS形式のforced-alignment labelを指定します。

#### `--font-file`

PNGで日本語を描画するフォントを明示します。

#### `--label-stride`

Boundary/Nucleus候補ラベルの表示間隔です。

```text
1: 全候補にラベル
5: 5候補ごとにラベル
```

Boundaryは`B001`、Nucleusは`N001`形式で表示されます。

#### `--minimum-token-confidence`

本文tokenとPhone列の最低alignment confidenceです。

```text
Default: 0.70
```

全体confidenceが閾値未満の場合、誤解を招く整列図を保存せず停止します。

### BASIC5000_0001の検証済み結果

```text
Transcript:
水をマレーシアから買わなくてはならないのです。

Aligned tokens:       12
Alignment confidence: 0.976
Frames:               639
Scales:                64
Boundary candidates:   74
Nucleus candidates:   102
Boundary-excluded:     31
```

### 生成HTMLの表示

```bash
.venv/bin/python -m http.server \
  8000 \
  --bind 127.0.0.1 \
  --directory "$UTTERANCE_DIR"
```

ブラウザーで次を開きます。

```text
http://127.0.0.1:8000/BASIC5000_0001_aligned.html
```

サーバーは`Ctrl+C`で終了します。

### 複数発話の可視化

```bash
cd "$HOME/projects/mrpa-II"

RESULT_ROOT="$HOME/projects/mrpa/data/jsut/results/mrpa-II/basic5000"
CORPUS_ROOT="$HOME/projects/mrpa/data/jsut/raw/jsut_ver1.1"
LAB_ROOT="$HOME/projects/mrpa-II/external/jsut-lab/basic5000/lab"
FONT_FILE="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

for number in $(seq -w 1 10)
do
  utterance_id="BASIC5000_00${number}"
  utterance_dir="${RESULT_ROOT}/${utterance_id}"
  lab_file="${LAB_ROOT}/${utterance_id}.lab"

  .venv/bin/python \
    scripts/visualize_mrpa_utterance.py \
    --input-dir "$utterance_dir" \
    --corpus-root "$CORPUS_ROOT" \
    --lab-file "$lab_file" \
    --font-file "$FONT_FILE" \
    --output "$utterance_dir/${utterance_id}_aligned.png" \
    --html-output "$utterance_dir/${utterance_id}_aligned.html" \
    --label-stride 5
done
```

### 既知の非ブロッキング警告

句読点など音素を生成しない要素に対して、OpenJTalkが次を表示する場合があります。

```text
WARNING: JPCommonLabel_make() in jcomon_label.c: No phoneme.
```

PNGとHTMLが保存され、alignment confidenceが表示されれば、処理自体は完了しています。

---

## 3. `generate_mrpa_index.py`

### 目的

複数発話の`*_aligned.png`と`*_aligned.html`を走査し、検索可能な一覧ページを生成します。

一覧ページには次が表示されます。

- 発話ID
- 日本語本文
- 縮小画像
- frame数
- scale数
- Boundary候補数
- Nucleus候補数
- Boundary近傍で除外された候補数
- 個別HTMLへのリンク
- 発話IDまたは本文による検索欄

### 一覧ページを生成

```bash
cd "$HOME/projects/mrpa-II"

RESULT_ROOT="$HOME/projects/mrpa/data/jsut/results/mrpa-II/basic5000"
TRANSCRIPT_FILE="$HOME/projects/mrpa/data/jsut/raw/jsut_ver1.1/basic5000/transcript_utf8.txt"

.venv/bin/python \
  scripts/generate_mrpa_index.py \
  --result-root "$RESULT_ROOT" \
  --transcript-file "$TRANSCRIPT_FILE" \
  --output "$RESULT_ROOT/index.html" \
  --title "MRPA-II basic5000 analysis" \
  --require-complete
```

期待される出力例:

```text
saved: .../basic5000/index.html
included: 10
skipped: 0
```

整列PNGまたはHTMLが未生成の発話は`skipped`として記録されます。

### 主なoption

#### `--result-root`

subset単位の結果rootです。

```text
例:
~/projects/mrpa/data/jsut/results/mrpa-II/basic5000
```

#### `--transcript-file`

対象subsetの`transcript_utf8.txt`です。

#### `--output`

一覧HTMLの保存先です。省略時は`<result-root>/index.html`です。

#### `--title`

一覧ページのタイトルです。

#### `--require-complete`

`.complete` markerを持つ発話だけを一覧候補にします。

### 一覧ページを表示

```bash
.venv/bin/python -m http.server \
  8000 \
  --bind 127.0.0.1 \
  --directory "$RESULT_ROOT"
```

ブラウザーで次を開きます。

```text
http://127.0.0.1:8000/index.html
```

---

## 推奨実行順序

```text
1. process_jsut.sh
   JSUT音声をMRPA-II成果物へ変換

2. visualize_mrpa_utterance.py
   各発話の整列PNG・HTMLを生成

3. generate_mrpa_index.py
   複数発話の検索可能な一覧HTMLを生成
```

## コミット対象と非対象

### コミット対象

```text
scripts/process_jsut.sh
scripts/visualize_mrpa_utterance.py
scripts/generate_mrpa_index.py
README_Scripts.md
```

### 通常はコミットしないもの

```text
external/jsut-lab/
data/
*.npy
*.npz
生成したPNG
生成したHTML
```

## 現在の研究上の位置付け

BoundaryとNucleusは、現段階では最終的な言語学的ラベルではなく、高感度の音響候補です。

```text
Interpolated F0
= 連続韻律場

Boundary candidates
= 韻律場の変化点候補

Nucleus candidates
= 局所prominence候補
```

次段階では、Boundaryを統合して可変長Prosodic Unitを構築し、各Unit内部でprimary Nucleusを0または1件選ぶ設計へ進みます。Phone、Mora、Vowelは、音響側で発見された構造を事後的に検証する参照情報として使用します。
