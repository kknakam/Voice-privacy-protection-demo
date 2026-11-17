# 日本語モデルと辞書

デモサイトの日本語のクローンに使う、学習済みのモデルと、音素化の辞書です。`manifest.json` が、ファイルの一覧、大きさ、sha256、版番号を持ちます。
1 ファイルが Cloudflare の上限 (25 MiB) に収まるよう、大きなファイルを 20 MiB ごとに分割しています (`*.part0`, `*.part1`, ...)。サイトが、順に取得して結合し、sha256 を確かめます。

| ファイル | 内容 |
|---|---|
| `ja_encoder.onnx` | 入力記号 -> エンコーダの出力と、デコーダの cross-attention の key / value |
| `ja_decoder_step.onnx` | デコーダの 1 ステップ分 (KV キャッシュの入出力、文の終わりの検出に使う注意 `align` を出力) |
| `ja_postnet_vocoder.onnx` | メルスペクトログラム -> postnet -> HiFi-GAN -> 波形 |
| `dict/` | OpenJTalk (NAIST Japanese Dictionary) の辞書 |

- **モデル:** Microsoft SpeechT5 (MIT) を、日本語の音素と韻律記号 (47 記号) に対応させて追加学習したもの。20,000 ステップ。重みは fp16 で保存し、計算は fp32
- **学習データ:** JVS コーパス (96 話者)、JSUT コーパス (basic5000)、LibriTTS の一部 (英語を忘れないため)
- **利用条件:** JVS と JSUT の音声の利用は、学術機関の研究、非営利の研究 (企業内を含む)、個人利用に限られ、再配布は認められていません。このモデルの利用は、その条件に従ってください。辞書は修正 BSD (`vendor/openjtalk/LICENSE-naist-jdic`)
- **版番号 (`manifest.json` の `version`):** モデルを差し替えるときは、必ず増やしてください。ブラウザに保存済みの古いデータが使われ続けるのを防ぎます
