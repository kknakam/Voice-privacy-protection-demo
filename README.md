# Voice-privacy-protection-demo

私が研究で取り組んだ手法を用いて簡易的なデモサイトを作成するリポジトリ

その場で録音した声から敵対的音声を生成し、音声クローンへの保護効果を確かめる静的サイトです。
すべての処理をブラウザ内で実行するため、配信サーバに GPU は不要で、音声は外部へ送信されません。

| 機能 | 内容 |
|---|---|
| 録音 | マイクから録音し、16 kHz モノラルへ変換して無音を除去、最長 8 秒に整形 |
| 敵対的音声の生成 | ターゲット話者（男性 / 女性）を選び、時間周波数マスクを最適化。反復回数は 50 / 100 / 200。各ターゲットの参照音声を試聴できる |
| クローン | 利用者が入力した英文を、録音と敵対的音声それぞれの話者埋め込みで SpeechT5 に合成させて比較 |

話者エンコーダは生成とクローンで同じ x-vector を使うホワイトボックス条件です。

## 構成

```
./
  index.html             画面
  _headers               Cloudflare Pages 用のレスポンスヘッダ (COOP/COEP, CSP)
  serve.py               ローカル確認用のサーバ (_headers を再現)
  css/style.css
  js/app.js              画面の制御
  js/recorder.js         録音と整形
  js/engine.js           ワーカーとの受け渡し
  js/engine.worker.js    重い処理の本体
  js/dsp.js              STFT / ISTFT / fbank の順伝播と随伴
  js/attack.js           マスク最適化 (損失と Adam)
  js/tts.js              入力文の正規化と SpeechT5 の生成ループ
  js/charts.js           スペクトログラムと波形の描画
  models/xvector_grad.onnx   x-vector の順伝播 + 勾配 (16.1 MiB)
  assets/constants.json  メルフィルタバンク、ターゲット話者の埋め込み、語彙、攻撃の設定
  assets/targets/        ターゲット話者の参照音声 (female.wav, male.wav。LibriTTS の発話)
  assets/icons.svg       Lucide のアイコン
  vendor/                ONNX Runtime Web, Inter
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

## 実装の考え方

ONNX Runtime Web には自動微分がありません。そこで勾配の計算を次の 2 つに分けています。

1. **x-vector の部分:** 順伝播と勾配 (解析的な VJP) を 1 つの ONNX にまとめる
2. **信号処理の部分:** マスクから fbank までの順伝播と随伴を `js/dsp.js` が FFT で計算する

PyTorch の autograd との相対誤差は、ONNX の勾配が 2.2×10⁻⁶、マスクから損失までの全連鎖が 1.8×10⁻⁶ です。

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

マルチスレッドに必要な COOP/COEP を設定できないため、GitHub Pages での配信は想定していません。
その場合もシングルスレッドで動作しますが、処理は遅くなります。

## 素材について

`models/xvector_grad.onnx`、`assets/constants.json`、`assets/targets/` は、研究用のコードから生成したものです。生成用のコードはこのリポジトリに含みません。

攻撃の設定（学習率、損失の重みなど）は `assets/constants.json` の `attack` にあります。Beta の β だけは、Beta 探索の採用値である 1.0 に揃えています。

## 制限

- SpeechT5 は英語専用です。録音は他の言語でも構いませんが、クローンの発話は英語になります
- 初回に約 189 MiB のモデルを取得します。PC のブラウザを推奨します
- SpeechT5 は推論時も dropout が有効なので、同じ入力でも合成結果と類似度は毎回変わります
- 効果はホワイトボックス条件のものです。他の音声合成モデルへの保護効果は示していません

## クレジット

- 話者エンコーダ: [SpeechBrain spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) (Apache-2.0)
- 音声合成: [Microsoft SpeechT5](https://huggingface.co/microsoft/speecht5_tts) / [HiFi-GAN](https://huggingface.co/microsoft/speecht5_hifigan) (MIT)、ONNX 変換は [Xenova/speecht5_tts](https://huggingface.co/Xenova/speecht5_tts)
- ターゲット話者の埋め込みと参照音声: LibriTTS (CC BY 4.0)。参照音声は話者 1066（女性）と 7553（男性）の発話を 16 kHz モノラルに変換し、音量を調整したもの
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (MIT)、[Inter](https://github.com/rsms/inter) (OFL-1.1)、[Lucide](https://lucide.dev/) (ISC)
