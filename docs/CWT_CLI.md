ここでは、CWT CLIを一度に完成させず、入力検証 → 信号変換 → scale生成 → CWT計算 → 保存 → CLI統合の順に独立してテストできるタスクへ分解します。各タスクは、前段の出力を確認してから次へ進める構成です。

実装対象
Plain Text
analysis/f0/extract_cwt.py
tests/test_extract_cwt.py

最終CLI：

Shell
python analysis/f0/extract_cwt.py \
--input <interpolated-f0.npy> \
--time-input <time-axis.npy> \
--voicing-input <voicing-mask.npy> \
--interpolation-mask-input <interpolation-mask.npy> \
--interpolation-metadata <interpolation-metadata.json> \
--output <cwt-output.npz> \
--metadata-output <cwt-metadata.json>
Task 0: 実行環境を確認する
目的

現在の PyWavelets が提供するCWT APIを確認し、使用可能なmethodとsignatureを固定します。

作業
Shell
python - <<'PY'
import inspect
import pywt
 
print("PyWavelets:", pywt.__version__)
print("cwt signature:", inspect.signature(pywt.cwt))
PY
確認項目
sampling_period
method
precision

がsignatureに含まれているか確認します。

完了条件
pywt.cwt をimportできる
インストール中のAPI signatureを記録できる
対応methodを決定できる
実装判断

初期実装では、互換性を優先してCLI選択肢を次に限定します。

Plain Text
conv
fft

auto は実際の環境で受理されることを確認できた場合のみ追加します。

Task 1: CLI parserを定義する
実装関数
Python
def build_parser() -> argparse.ArgumentParser:
...
必須引数
Plain Text
--input
--time-input
--output
任意の関連入力
Plain Text
--voicing-input
--interpolation-mask-input
--interpolation-metadata
--metadata-output
信号変換パラメータ
Plain Text
--domain
--centering
--scaling

初期選択肢：

Plain Text
--domain log-hz|hz
--centering median|mean|none
--scaling none

standard と robust は次の比較段階へ送ります。

CWTパラメータ
Plain Text
--wavelet
--method
--precision
--number-of-scales
--minimum-period
--maximum-period
--scale-spacing

既定値：

Plain Text
domain = log-hz
centering = median
scaling = none
wavelet = morl
method = conv
precision = 12
number-of-scales = 64
minimum-period = 0.05
maximum-period = None
scale-spacing = log

maximum-period=None の場合：

pmax⁡=min⁡(2.0, D/2)p_{\max} = \min(2.0,\ D/2)pmax​=min(2.0, D/2)

ここで DDD は入力系列の継続時間です。

完了条件
Shell
python analysis/f0/extract_cwt.py --help
 

で全引数が表示される。

Task 2: パスと引数を検証する
実装関数
Python
def validate_arguments(args: argparse.Namespace) -> None:
...
検証内容
入力ファイル
必須ファイルが存在する
通常ファイルである
配列入力は .npy
metadataは .json
出力ファイル
CWT出力は .npz
metadata出力は .json
出力パス同士が重複しない
数値引数
Plain Text
precision >= 12
number_of_scales >= 2
minimum_period > 0
maximum_period > minimum_period
0 <= max_interpolation_ratio <= 1
文字列引数

選択肢外の値は argparse で拒否する。

テスト
存在しない入力
不正拡張子
負のperiod
scale数が1
precisionが12未満
出力パス重複
完了条件

不正設定がCWT計算前にすべて停止する。

Task 3: F0と時間軸をロードする
実装関数
Python
def load_core_inputs(
f0_path: Path,
time_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
...
処理
Python
np.load(path, allow_pickle=False)

でロードし、float64 へ変換する。

検証
両方1次元
shape一致
16フレーム以上
finite
F0はすべて正
時間軸は厳密単調増加
失敗ケース
Plain Text
NaN
Inf
0 Hz
negative F0
empty input
shape mismatch
non-increasing time
fewer than 16 frames
完了条件

正常な入力について、

Python
f0.dtype == np.float64
time_axis.dtype == np.float64
f0.shape == time_axis.shape

が成立する。

Task 4: Sampling periodを推定する
実装関数
Python
def infer_sampling_period(
time_axis: np.ndarray,
*,
rtol: float = 1e-6,
atol: float = 1e-9,
) -> float:
...
計算
Python
differences = np.diff(time_axis)
sampling_period = float(np.median(differences))

等間隔性：

Python
np.allclose(
differences,
sampling_period,
rtol=rtol,
atol=atol,
)
出力
Python
sampling_period_seconds
sampling_frequency_hz = 1.0 / sampling_period_seconds
テスト
正常
Python
[0.000, 0.005, 0.010, 0.015]

期待値：

Plain Text
sampling period = 0.005
sampling frequency = 200.0
異常
Python
[0.000, 0.005, 0.011, 0.016]

明示的に拒否する。

完了条件

F0抽出時の5 msフレーム周期を正しく復元できる。

Task 5: Maskをロードして検証する
実装関数
Python
def load_optional_mask(
path: Path | None,
*,
expected_shape: tuple[int, ...],
name: str,
) -> np.ndarray | None:
...
要件
未指定なら None
指定時は1次元
F0とshape一致
dtypeは bool

整数の0/1を暗黙にboolへ変換しない方針とします。

追加検証

両maskが存在する場合：

Python
mask_overlap = np.count_nonzero(
voicing_mask & interpolation_mask
)

初期パイプラインではoverlapは0であるべきです。

また、

Python
mask_complementary = np.array_equal(
interpolation_mask,
~voicing_mask,
)

を計算してmetadataへ記録します。

完了条件

maskを変形せず、後段の .npz へ引き継げる。

Task 6: Interpolation metadataを検証する
実装関数
Python
def load_interpolation_metadata(
path: Path | None,
) -> dict[str, Any] | None:
...
 
Python
def validate_interpolation_metadata(
metadata: dict[str, Any],
*,
number_of_frames: int,
voicing_mask: np.ndarray | None,
interpolation_mask: np.ndarray | None,
) -> None:
...
 
必須整合条件
Python
metadata["total_frames"] == number_of_frames

mask指定時：

Python
metadata["observed_frames"] == np.count_nonzero(voicing_mask)
metadata["interpolated_frames"] == np.count_nonzero(
interpolation_mask
)

さらに、

Python
metadata["observed_values_preserved"] is True

を要求します。

完了条件

入力F0と、その生成条件を結び付けられる。

Task 7: CWT入力信号を生成する
実装関数
Python
def prepare_cwt_signal(
continuous_f0: np.ndarray,
*,
domain: str,
centering: str,
scaling: str,
) -> tuple[np.ndarray, dict[str, float | str]]:
...
変換順序
Plain Text
Continuous F0
↓
Domain conversion
↓
Centering
↓
Scaling
↓
CWT input signal
Domain conversion
log-hz
Python
signal = np.log(continuous_f0)
hz
Python
signal = continuous_f0.copy()
Centering
median
Python
center = float(np.median(signal))
signal = signal - center
mean
Python
center = float(np.mean(signal))
signal = signal - center
none
Python
center = 0.0
Scaling

初期実装：

Plain Text
none

のみ。

Dynamic range
Python
dynamic_range = float(np.ptp(signal))

次の場合は拒否：

Python
dynamic_range <= 1e-12
返却情報
Python
{
"domain": domain,
"centering": centering,
"scaling": scaling,
"center_value": center,
"dynamic_range": dynamic_range,
}
完了条件
log-F0変換が正しい
median centering後のmedianが0に近い
元のF0配列を変更しない
定数入力を拒否する
Task 8: 解析period範囲を決定する
実装関数
Python
def resolve_period_range(
*,
minimum_period: float,
maximum_period: float | None,
sampling_period: float,
duration: float,
) -> tuple[float, float]:
...
最小period

要件：

Python
minimum_period > sampling_period

より安全には、最低4サンプルを要求します。

Python
minimum_period >= 4.0 * sampling_period

5 ms入力なら：

Plain Text
minimum period >= 20 ms

既定値はprosody-focusedとして：

Plain Text
50 ms
最大period

未指定時：

Python
maximum_period = min(
2.0,
duration / 2.0,
)
失敗条件
Plain Text
maximum_period <= minimum_period
maximum_period > duration / 2

後者は当面エラーとし、暗黙clampはしません。自動値の場合だけclampします。

完了条件

解析範囲が入力長と矛盾しない。

Task 9: Periodからscaleを生成する
実装関数
Python
def generate_scales(
*,
wavelet: str,
sampling_p*riod: float,
minimum_period: f*oat,
maximum_period: float,
* number_of_scales: int,
spacin*: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
...
処理手順
1. period列を生成

対数配置：

Python
periods = np.geomspace(
* minimum_period,
maximum_perio*,
number_of_scales,
)

線形配置：

Python
periods = np.linspace*
minimum_period,
maximum_p*riod,
number_of_scales,
)
2. 物理周波数へ変換
Python
frequen*ies_hz = 1.0 / periods
3. normalized frequencyへ変換
Python
normalized_frequencies = (
fre*uencies_hz * sampling_period
)
```*
4. scaleへ変換
Python
scales = pywt.frequency2scale(
wavele*,
normalized_frequencies,
)
``*
5. 実scaleから周波数を再計算
Python
actual_frequencies_hz = (
pyw*.scale2frequency(wavelet, scales)
* / sampling_period
)
 
actual_per*ods_seconds = 1.0 / actual_frequen*ies_hz
必須検証
全scaleがfinite
全scaleが正
shapeがscale数と一致
frequencyが正
periodが正
periodが昇順
frequencyが降順
完了条件

scale番号ではなく、物理periodからCWT解析範囲を構築できる。

Task 10: CWTを計算する
実装関数
Python
def compute_cwt(
signal: np.n*array,
*,
scales: np.ndarr*y,
wavelet: str,
sampling_*eriod: float,
method: str,
*precision: int,
) -> tuple[np.ndarray, np.ndarray]:
...
呼び出し
Python
coefficients, freque*cies_hz = pywt.cwt(
signal,
* scales,
wavelet,
sampling*period=sampling_period,
method*method,
precision=precision,
)*
検証
Python
coefficients*shape == (
scales.size,
si*nal.size,
)
 
frequencies_hz.shape *= scales.shape

さらに：

Python
np.all(np.isfinite(coefficients))*np.all(np.isfinite(frequencies_hz)*
np.all(frequencies_hz > 0.0)

複素waveletに備えて、係数を無条件に float64 へ変換しないようにします。

完了条件

実waveletと複素waveletのどちらでも、係数dtypeを保存できる。

Task 11: 品質情報を計算する
実装関数
Python
def summarize_input_quality(
* *,
number_of_frames: int,
*voicing_mask: np.ndarray | None,
* interpolation_mask: np.ndarray |*None,
max_interpolation_ratio:*float,
) -> dict[str, Any]:
..*
interpolation ratio
Python
ratio = (
np.count_nonzer*(interpolation_mask)
/ number_*f_frames
)
quality flag
Plain Text
ratio <= 0.20 low
0.20 < ratio <= 0.50 moderate
ratio > 0*50 high

境界値の意味が曖昧にならないよう、コードとmetadataで定義を固定します。

拒否条件
Python
ratio > max_interpolatio*_ratio

の場合はエラー。

mask未指定の場合：

Plain Text
interpolation_ratio = null*interpolation_quality = unknown
``*
完了条件

CWT係数と補間依存度を一緒に評価できる。

Task 12: NPZ出力を保存する
実装関数
Python
def save_cwt_archive(
* output_path: Path,
*,
co*fficients: np.ndarray,
scales:*np.ndarray,
frequencies_hz: np*ndarray,
periods_seconds: np.n*array,
time_axis: np.ndarray,
* input_signal: np.ndarray,
c*ntinuous_f0_hz: np.ndarray,
vo*cing_mask: np.ndarray | None,
*nterpolation_mask: np.ndarray | No*e,
) -> None:
...
必須キー
Plain Text
coefficients
scales
frequ*ncies_hz
periods_seconds
time_axis*input_signal
continuous_f0_hz
Maskの扱い

NPZ内でキーを常に固定するなら、未指定時は空bool配列を保存します。

Python
voicing_m*sk=np.array([], dtype=bool)
interp*lation_mask=np.array([], dtype=boo*)

併せてmetadataへ、

Plain Text
voic*ng_mask_available
interpolation_ma*k_available

を保存します。

安全な保存

途中失敗で不完全ファイルを残さないため、可能なら一時ファイルへ保存してからrenameします。

完了条件

保存後に再ロードして全配列を復元できる。

Task 13: JSON metadataを構築する
実装関数
Python
def*build_metadata(...) -> dict[str, A*y]:
...
セクション構成
JSON
{
"inputs": {},
"signal": {}*
"sampling": {},
"scales": {},* "cwt": {},
"quality": {},
"e*ge_effects": {}
}
記録内容
inputs
入力ファイル
maskファイル
interpolation metadata
出力ファイル
signal
domain
centering
center value
scaling
dynamic range
sampling
frame count
duration
sampling period
sampling frequency
scales
number of scales
spacing
minimum period
maximum period
minimum frequency
maximum frequency
cwt
wavelet
method
precision
coefficient dtype
coefficient shape
quality
voiced ratio
interpolation ratio
interpolation quality
short-input warning
mask complementarity
edge effects

初期実装：

JSON
{
"padding": "none",
"cone_of_influence_available": false,
"edge_effect_warning": true
}
完了条件

NPZを開かなくても、実験条件をJSONから確認できる。

Task 14: main関数で統合する
実装関数
Python
def main() -> int:
...
処理順序
Plain Text
1. Parse arguments
2. Validate paths and parameters
3. Load F0 and time
4. Infer sampling period
5. Load masks
6. Validate interpolation metadata
7. Prepare CWT signal
8. Resolve period range
9. Generate scales
10. Compute CWT
11. Summarize quality
12. Save NPZ
13. Save metadata
14. Print summary
エラー処理

捕捉対象：

Python
FileNotFoundError
OSError
ValueError
RuntimeError
json.JSONDecodeError

表示：

Plain Text
error: <具体的理由>

終了コード：

Plain Text
success: 0
failure: 1
成功時の表示
Plain Text
input F0 : ...
input time : ...
frames : ...
sampling period : ...
sampling frequency : ...
domain : ...
centering : ...
wavelet : ...
method : ...
scales : ...
period range : ...
coefficient shape : ...
saved CWT : ...
saved metadata : ...
Task 15: Unit testsを実装する

tests/test_extract_cwt.py を作成します。

テスト群
入力検証
正常入力
F0 shape不正
時間shape不正
shape不一致
F0の0
F0の負値
NaN
Inf
非単調時間軸
不等間隔時間軸
16フレーム未満
信号変換
log-hz変換
hz変換
median centering
mean centering
none centering
元配列を変更しない
定数信号を拒否
Scale
scale数一致
全scale正
log spacing
linear spacing
period-frequency逆数関係
period範囲
CWT
coefficient shape
frequency shape
全係数finite
実waveletdtype
metadata整合
Mask
bool maskを受理
非bool maskを拒否
shape不一致を拒否
interpolation ratio
最大補間率超過を拒否
Task 16: 既知変調信号の統合テスト
テスト入力

フレーム周期：

Python
dt = 0.005

継続時間：

Python
duration = 4.0

F0：

f(t)=220exp⁡(0.1sin⁡(2π⋅2t))f(t) = 220\exp(0.1\sin(2\pi\cdot2t))f(t)=220exp(0.1sin(2π⋅2t))
手順
F0を生成
time axisを生成
log-hz変換
median center
CWT
scaleごとの平均powerを計算
Python
power_by_scale = np.mean(
np.abs(coefficients) ** 2,
axis=1,
)
最大powerのfrequencyを取得
受け入れ条件

主要frequencyが2 Hz付近にあることを確認します。

初期許容範囲：

Plain Text
1.5 Hz <= peak frequency <= 2.5 Hz

この範囲はscale密度64との組み合わせでテストし、必要なら許容幅を実測に基づいて調整します。

Task 17: CLI smoke test

実際の補間済み220 Hzファイルはほぼ定数なので、CWTの統合確認にはF0変調データを作ります。

Shell
mkdir -p data/test/cwt

生成、CLI実行、NPZ再ロードを一連で検証します。

完了条件
.npz が生成される
JSON metadataが生成される
coefficient shapeが (64, T)
全係数がfinite
frequencyとperiodが整合
入力信号が保存されている
Task 18: 全テストとコミット
構文確認
Shell
python -m py_compile \
analysis/f0/extract_cwt.py \
tests/test_extract_cwt.py
CWTテスト
Shell
python -m pytest tests/test_extract_cwt.py -q
全テスト
Shell
python -m pytest -q
コミット
Shell
git add \
analysis/f0/extract_cwt.py \
tests/test_extract_cwt.py
 
git diff --cached --check
git diff --cached --stat
 
git commit -m "Add tested continuous wavelet transform pipeline"
git push origin main
実装順序

実装を次の6マイルストーンで進めます。

Milestone 1: 入力契約
Task 1から6
F0、時間軸、mask、metadata検証
Milestone 2: 信号変換
Task 7
log-F0とcentering
Milestone 3: Physical Scale Assignment
Task 8から9
periodからscaleを生成
Milestone 4: CWT Core
Task 10から11
CWT計算と品質情報
Milestone 5: Reproducible Output
Task 12から14
NPZ、JSON、CLI
Milestone 6: Verification
Task 15から18
単体テスト、既知周波数テスト、CLI smoke test

最初の実装では、機能を増やしすぎず、log-hz + median centering + morl + physical-period scales + no padding を再現可能なbaselineとして完成させます。