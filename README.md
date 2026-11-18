# Voice-privacy-protection-demo

私が研究で取り組んだ手法を用いて簡易的なデモサイトを作成するリポジトリ

その場で録音した声から敵対的音声を生成し、音声クローンへの保護効果を確かめる静的サイトです。
すべての処理をブラウザ内で実行するため、配信サーバに GPU は不要で、音声は外部へ送信されません。

| 機能 | 内容 |
|---|---|
| 録音 | マイクから録音し、16 kHz モノラルへ変換して無音を除去、最長 8 秒に整形 |
| 敵対的音声の生成 | ターゲット話者（男性 / 女性）を選び、時間周波数マスクを最適化。反復回数は 50 / 100 / 200。各ターゲットの参照音声を試聴できる |
| クローン | 利用者が入力した英文を、録音と敵対的音声それぞれの話者埋め込みで SpeechT5 に合成させて比較 |
| 日本語のクローン | 日本語に追加学習した SpeechT5 で、日本語の文章 (定型文または自由入力) を同様に合成して比較。モデルと辞書は同梱 |
| 研究のシナリオ | 別のタブ (URL の `#scenario` で直接開ける) で、研究が想定する場面、守る仕組み、敵対的音声のつくり方、デモとの対応、結果の読み方、条件と限界を、図で説明。デモの録音や結果は、タブを切り替えても残る |

話者エンコーダは生成とクローンで同じ x-vector を使うホワイトボックス条件です。

## 構成

```
./
  index.html             画面
  _headers               Cloudflare Pages 用のレスポンスヘッダ (COOP/COEP, CSP)
  serve.py               ローカル確認用のサーバ (_headers を再現)
  css/style.css
  js/app.js              画面の制御
  js/tabs.js             「デモ」と「研究のシナリオ」のタブ (ARIA の tabs。URL のハッシュで開く)
  js/scenario.js         研究のシナリオの図 3 (マスクの掛け算の模式図) を描く
  js/recorder.js         録音と整形
  js/engine.js           ワーカーとの受け渡し
  js/engine.worker.js    重い処理の本体
  js/dsp.js              STFT / ISTFT / fbank の順伝播と随伴
  js/attack.js           マスク最適化 (損失と Adam)
  js/tts.js              入力文の正規化と SpeechT5 の生成ループ
  js/tts_ja.js           日本語モデルの生成ループ (文の終わりの検出を含む)
  js/ja_frontend.js      日本語の音素化 (OpenJTalk の WASM と、アクセント結合前の規則)
  js/ja_g2p.js           フルコンテキストラベルから、音素と韻律記号の ID 列を作る
  js/charts.js           スペクトログラムと波形の描画
  models/xvector_grad.onnx   x-vector の順伝播 + 勾配 (16.1 MiB)
  assets/constants.json  メルフィルタバンク、ターゲット話者の埋め込み、語彙、攻撃の設定
  assets/targets/        ターゲット話者の参照音声 (female.wav, male.wav。LibriTTS の発話)
  assets/icons.svg       Lucide のアイコン
  vendor/                ONNX Runtime Web, Inter, OpenJTalk (WASM)
  models_ja/             日本語モデルと辞書 (20 MiB ごとに分割。manifest.json に大きさと sha256)
```

SpeechT5 のモデルは同梱せず、初回に Hugging Face からコミット固定で取得してブラウザに保存します。
Cloudflare Pages の 1 ファイル上限 (25 MiB) を超えるためです。

| ファイル | サイズ | 理由 |
|---|---|---|
| `encoder_model_bnb4.onnx` | 48.4 MiB | 量子化しても声の再現性と発音が fp32 と同等 |
| `decoder_model_merged_fp16.onnx` | 114.0 MiB | 量子化すると声の再現性が大きく落ちるため fp16 |
| `decoder_postnet_and_vocoder_fp16.onnx` | 26.5 MiB | 同上 |

デコーダを量子化すると、クローンと本人の類似度は fp16 の 0.941 から 0.860 まで下がります
(LibriTTS 6 話者、各 3 回の平均。異話者の平均は 0.899)。

## 日本語のクローン

日本語は、Microsoft SpeechT5 を日本語に追加学習したモデルで合成します。英語のモデルとは別に、`models_ja/` に同梱しています。

| 項目 | 内容 |
|---|---|
| 入力 | 日本語の文章を OpenJTalk で音素と韻律記号 (アクセント句の境界、ピッチの上昇・下降など) にしたもの。ブラウザ内で、WebAssembly に変換した OpenJTalk を使う |
| 話者条件 | 英語と同じ x-vector (話者エンコーダは生成とクローンで共通) |
| モデル | encoder 178.3 MiB、decoder 103.0 MiB、vocoder 26.5 MiB (重みだけ fp16、計算は fp32)。辞書 102 MiB。20 MiB ごとに分割して置き、取得後に大きさと sha256 を確かめる |
| 学習データ | JVS コーパス (96 話者)、JSUT コーパス (basic5000)、英語を忘れないための LibriTTS の一部。計 約 33 時間、20,000 ステップ。JVS の 4 話者は学習に使わず、評価用にした |
| 初回の取得 | 約 308 MiB (定型文のみ)。自由入力では、さらに辞書の約 102 MiB |

評価 (学習に使っていない 12 話者 x 15 文。Whisper large-v3 による文字誤り率、x-vector による話者類似度):

| | 文字誤り率 (表記) | 話者類似度 |
|---|---|---|
| 自然音声 (JSUT、同じ文) | 0.8% | — |
| 日本語モデルの合成 | 1.4% | 0.948 |

- デコーダが同じ文を 2 回読んでしまう失敗が、JVS の声で起きやすいため、デコーダの注意から文の終わりを検出して止めています (止めない場合、JVS の未知話者の文字誤り率は 16.6%)
- 自由入力の音素化は、OpenJTalk の素の版です。学習時の音素化 (Sudachi による読みの補正などを含む) とは、約 4 分の 1 の文で記号列が違います。聞き取りへの影響は小さいと確認していますが、アクセントが不自然になることがあります
- 評価は少数の話者と文で行ったものです。アクセントの自然さは、文字誤り率では測れていません

## 実装の考え方

ONNX Runtime Web には自動微分がありません。そこで勾配の計算を次の 2 つに分けています。

1. **x-vector の部分:** 順伝播と勾配 (解析的な VJP) を 1 つの ONNX にまとめる
2. **信号処理の部分:** マスクから fbank までの順伝播と随伴を `js/dsp.js` が FFT で計算する

PyTorch の autograd との相対誤差は、ONNX の勾配が 2.2×10⁻⁶、マスクから損失までの全連鎖が 1.8×10⁻⁶ です。

## 研究のシナリオ (タブ)

- **タブ:** `js/tabs.js` が、「デモ」と「研究のシナリオ」のパネルを `hidden` で切り替えます。パネルは消さないので、録音、生成結果、入力した文章は、タブを往復しても残ります。タブごとにスクロール位置を覚え、再生中の音声は、タブを離れると一時停止します。処理中は、「デモ」のタブに印が出ます
- **URL:** `#scenario` で直接開けます。`#step-clone` のような、デモ内の要素を指すハッシュでも、タブを切り替えてその位置へ移動します。戻る・進むにも対応します。`app.js` とは独立なので、エンジンの読み込みが失敗しても、タブは使えます
- **図:** 流れの図は HTML と CSS だけで描きます (1040px 以上で横並び、それより狭いと縦並び)。「加工する / しない」の切り替えはラジオボタンと CSS (`:has()`) だけで動きます。図 3 の格子は `js/scenario.js` が、図 6 の模式図は SVG で描きます。CSP (`style-src 'self'`) に合わせて、インラインの style と script は使いません
- **色:** 紫 = 本人、青緑 = 保護、橙 = 攻撃者 (クローン)。各カードに、役割を文字でも書いています
- **内容:** タブは、研究の想定場面と仕組みの解説に絞っています。検証の数値は載せず、下の「ブラウザでの検証結果」に書いています。図 6 の位置や距離は、模式図で、実際の値ではありません
- **アイコン:** [Lucide](https://lucide.dev/) 1.74.0 の SVG を、`assets/icons.svg` に取り込んでいます

## ブラウザでの検証結果

Chromium の疑似マイクに LibriTTS の音声を入力し、画面操作で全工程を通した結果です
(5 話者、各 8 秒、100 反復、利用者入力を想定した 5 種類の英文)。

| クローンの元 | 本人との類似度 | ターゲットとの類似度 | FSR | WER |
|---|---|---|---|---|
| 録音 (保護なし) | 0.962 | 0.880 | 0% | 0.062 |
| 敵対的音声 (保護あり) | 0.878 | 0.955 | 100% | 0.077 |

- 画面に表示された類似度は、保存した音声を PyTorch の x-vector で測り直した値と一致しました
- この検証は、ターゲット話者を選択式にする前の版（録音から遠い話者を自動で選ぶ）で実施したものです。選択式の現行版での再測定は未実施です
- WER の差の主因は、Whisper が "twenty-five dollars" を "$25" と書き起こすなどの表記揺れです
- 速度は 216 ms/ステップ (8 秒の音声、WASM 4 スレッド、16 コアのサーバ)。一般的な PC では遅くなります

実際のマイク録音、Firefox / Safari、一般的な PC での速度は未検証です。

## ローカルで動かす

COOP/COEP が付かないとシングルスレッドになるので、`_headers` を再現するサーバを使います。

```bash
python serve.py                              # http://localhost:8000/
```

マイクは `localhost` か HTTPS でのみ使えます。

## 公開する (GitHub + Cloudflare Pages)

ビルドは不要です。Cloudflare Pages のビルドコマンドは空にし、ビルド出力先を `/` にします。

`models_ja/` の分割ファイルは、1 つが 20 MiB で、Cloudflare の 1 ファイル 25 MiB の上限に収まります。

マルチスレッドに必要な COOP/COEP を設定できないため、GitHub Pages での配信は想定していません。
その場合もシングルスレッドで動作しますが、処理は遅くなります。

## 素材について

`models/xvector_grad.onnx`、`assets/constants.json`、`assets/targets/` は、研究用のコードから生成したものです。生成用のコードはこのリポジトリに含みません。

攻撃の設定（学習率、損失の重みなど）は `assets/constants.json` の `attack` にあります。Beta の β だけは、Beta 探索の採用値である 1.0 に揃えています。

## 制限

- 英語のモデル (SpeechT5) は英語専用です。録音は他の言語でも構いません。日本語の発話は、日本語のモデルを選んだときだけ合成されます
- 初回に、英語で約 189 MiB、日本語で約 308 MiB (自由入力ではさらに辞書の約 102 MiB) を取得します。日本語は、ブラウザのメモリも多く使います (開発環境の実測で 1.2〜1.5 GB 程度)。PC のブラウザを推奨します
- 日本語のモデルは、JVS の声に近い x-vector で、短い文を、同じ文の繰り返しにしてしまうことがあります。終わりの検出で止めていますが、注意が崩れて末尾に届かない場合には効きません
- SpeechT5 は推論時も dropout が有効なので、同じ入力でも合成結果と類似度は毎回変わります
- 効果はホワイトボックス条件のものです。他の音声合成モデルへの保護効果は示していません
- 日本語のクローンでの保護効果は、系統立てた評価をしていません (評価したのは、日本語モデルの合成品質です)

## クレジット

- 話者エンコーダ: [SpeechBrain spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) (Apache-2.0)
- 音声合成: [Microsoft SpeechT5](https://huggingface.co/microsoft/speecht5_tts) / [HiFi-GAN](https://huggingface.co/microsoft/speecht5_hifigan) (MIT)、ONNX 変換は [Xenova/speecht5_tts](https://huggingface.co/Xenova/speecht5_tts)
- ターゲット話者の埋め込みと参照音声: LibriTTS (CC BY 4.0)。参照音声は話者 1066（女性）と 7553（男性）の発話を 16 kHz モノラルに変換し、音量を調整したもの
- 日本語の音素化: [Open JTalk](https://open-jtalk.sourceforge.net/) と NAIST Japanese Dictionary (いずれも修正 BSD)。WebAssembly に変換して使用。ライセンスは `vendor/openjtalk/` に同梱
- 日本語モデルの学習データ: [JVS コーパス](https://arxiv.org/abs/1908.06248) と [JSUT コーパス](https://arxiv.org/abs/1711.00354) (東京大学)、LibriTTS の一部 (CC BY 4.0)。**JVS と JSUT の音声の利用は、学術機関の研究、非営利の研究 (企業内を含む)、個人利用に限られ、再配布は認められていません。** このデモの日本語モデルは、その条件に従い、非営利の研究・個人利用に限って使ってください
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (MIT)、[Inter](https://github.com/rsms/inter) (OFL-1.1)、[Lucide](https://lucide.dev/) (ISC)
