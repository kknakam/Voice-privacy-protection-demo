# Voice-privacy-protection-demo

私が研究で取り組んだ手法を用いて簡易的なデモサイトを作成するリポジトリ

## 概要

その場で録音した声から敵対的音声を生成し、音声クローンへの保護効果を確かめるサイトです。

| Step | 内容 | 処理する場所 |
|---|---|---|
| 1 録音 | マイクから録音し、16 kHz モノラル、最長 8 秒に整形 | ブラウザ |
| 2 敵対的音声の生成 | 録音の話者からいちばん遠い話者を、候補 24 人から選び、時間周波数マスクを最適化。「残した情報の割合」、SNR、UTMOS を表示 | サーバ (Python、GPU) |
| 3 クローンの比較 | 入力した文章を、録音と敵対的音声それぞれの話者埋め込みで、SpeechT5 に合成させて比較。ひらがな・カタカナ・漢字が 1 字でもあれば日本語 (追加学習した SpeechT5)、なければ英語 | サーバ |

別のタブ「研究のシナリオ」(URL の `#scenario`) で、研究が想定する場面と、守る仕組みを図で説明します。

- 録音と入力した文章は、処理が終わるとすぐに、サーバから削除します。
- 生成には、研究のリポジトリ (AntiCloning_mask) のコードを使います。このリポジトリには含みません。
- 効果は、話者エンコーダが生成とクローンで共通の、ホワイトボックス条件のものです。他の音声合成モデルへの効果は、示していません。

## 構成

```
content/        タブの本文 (MDX): scenario.mdx (研究のシナリオ)、demo.mdx (デモ)
tools/          scenario/: MDX から HTML を作る。serve.py: サイトだけを配信する簡易サーバ
site/           配信する部分: index.html、_headers、css/、js/ (demo/ が画面の制御と録音)、assets/、vendor/
server/         サーバ (Python)
  app.py          サイトと API の配信。順番待ち。録音と文章は処理後に削除
  core.py         敵対的音声の生成と測定 (研究のコードを import)
  tts.py          音声合成 (英語: PyTorch、日本語: ONNX Runtime)
  models_ja/      日本語の音声合成のモデル
  data/           候補話者 24 人の x-vector (pool.npz)、音声合成の語彙 (tts.json)
  env.example     環境設定の書き方
scripts/        実行用のシェルスクリプト (setup_server.sh、start.sh / stop.sh / status.sh、serve.sh、check_env.sh など)
deploy/         サーバへ置くための雛形 (systemd、nginx、Cloudflare Tunnel)
requirements.txt  サーバの Python の依存パッケージ (版を固定)
package.json      MDX から HTML を作るのに使う (サイトの配信には使わない)
```

## 本文を編集する (MDX)

2 つのタブの本文は、Markdown と部品 (JSX) で書きます。編集したら、次を実行して `site/index.html` に反映します (Node.js 22 以上)。

| タブ | 本文 | 部品の定義 |
|---|---|---|
| 研究のシナリオ | `content/scenario.mdx` | `tools/scenario/components.mjs` |
| デモ | `content/demo.mdx` | `tools/scenario/demo-components.mjs` |

```bash
npm install                  # 初回だけ
npm run build                # 2 つの MDX から HTML を生成し、site/index.html の印の間を書き換える
npm run check                # 書き換えずに、index.html が MDX と一致しているかを確かめる
npm run build:scenario       # シナリオだけ (build:demo は、デモだけ)
```

- `site/index.html` の `BEGIN ...-content` と `END ...-content` の間は自動生成です。手で直しても、次の生成で消えます。見出し画面とフッターは、手書きです。
- 書き間違い (存在しない図への参照、足りない項目など) は、`エラー: …` の 1 行で、MDX の行番号とともに知らせます。
- デモの文章は自由に書き換えられますが、`record-start` のような **id は変えられません**。`site/js/demo/app.js` が、部品を探す名前です。ボタンや入力欄を増やす場合は、`app.js` の変更も要ります。進み具合やエラーなど、動作中に出る文は、`app.js` にあります。
- シナリオの図のカードには、絵を出せます。絵を `site/assets/illust/` (無ければ作ります) に置き (SVG を推奨、1 枚 300 KiB 以下、縦横比 4:3、背景は透明、文字は入れない)、`<Step>` に `image="ファイル名" alt="絵の説明"` を足します (`alt` は必須)。

## サーバを起動する

マイクは、`http://localhost:ポート番号/` か `https://` の URL でしか使えません。

**サイトだけ見る場合** (Step 2 と Step 3 は使えません):

```bash
python tools/serve.py                        # http://localhost:8000/
```

**Step 2 と Step 3 も使う場合:** Linux (x86_64)、NVIDIA の GPU、Python 3.10 (3.9〜3.12)、研究のリポジトリ、重みが要ります。重みは、実行時にネットから取らないので、あらかじめ置きます。

初回は、次のコマンドで、仮想環境、パッケージ、環境ファイル、重みのコピー、通しの点検まで、自動で行います (約 6 分)。`--dry-run` を付けると、実行する内容だけを表示します。GPU がなければ、理由を示して止まります。

```bash
scripts/setup_server.sh \
  --mask-root /path/to/AntiCloning_mask \
  --from-hf   元のホスト:/path/to/hf \
  --from-hub  元のホスト:/path/to/torch_hub \
  --smoke
```

手作業で行う場合は、`requirements.txt`、`server/env.example`、`scripts/copy_weights.sh`、`scripts/check_env.sh` の冒頭の説明を参照してください。

起動と停止:

```bash
scripts/start.sh 8000        # バックグラウンドで起動。音声合成のモデルの読み込みに、約 20 秒かかる
scripts/status.sh            # 応答が "tts": {"state": "ready", ...} になれば、使える
scripts/stop.sh              # 止める (動いている間は、GPU とメモリを使い続ける)
scripts/serve.sh 8000        # 前面で動かす場合
```

**離れたサーバで動かし、手元の PC で見る場合:** サーバは、`127.0.0.1` だけで待ち受けます。手元の PC へ、ポートを転送し、`http://localhost:8000/` を開きます (VS Code の Remote-SSH の「ポート」でも同じです)。

```bash
ssh -L 8000:127.0.0.1:8000 ユーザー名@サーバ名
```

- サーバの IP アドレスやホスト名の `http://` の URL では、ブラウザがマイクを止めるため、録音できません。
- ポート転送では、接続元が全員 `127.0.0.1` になり、同時に処理できるのは 1 人です。各自の端末から、同時に使うには、HTTPS の入口が要ります (`deploy/nginx.example.conf`、または `scripts/tunnel.sh` の Cloudflare Tunnel)。
- 認証はありません。研究室の外へ公開する場合は、アクセスの制限と、フッターのプライバシーの文面を、確認してください。

## クレジット

- 話者エンコーダ: [SpeechBrain spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) (Apache-2.0)
- 音声合成: [Microsoft SpeechT5](https://huggingface.co/microsoft/speecht5_tts) / [HiFi-GAN](https://huggingface.co/microsoft/speecht5_hifigan) (MIT)。日本語は、SpeechT5 を追加学習したモデル
- ターゲット話者の埋め込みと参照音声: LibriTTS (CC BY 4.0)。参照音声は、候補 24 人の発話を 16 kHz モノラルに変換し、音量を調整したもの
- 日本語の音素化: [Open JTalk](https://open-jtalk.sourceforge.net/) と NAIST Japanese Dictionary (いずれも修正 BSD)。[pyopenjtalk-plus](https://pypi.org/project/pyopenjtalk-plus/) 経由で、サーバが使用
- 日本語モデルの学習データ: [JVS コーパス](https://arxiv.org/abs/1908.06248) と [JSUT コーパス](https://arxiv.org/abs/1711.00354) (東京大学)、LibriTTS の一部 (CC BY 4.0)。**JVS と JSUT の音声の利用は、学術機関の研究、非営利の研究 (企業内を含む)、個人利用に限られ、再配布は認められていません。** このデモの日本語モデルは、その条件に従い、非営利の研究・個人利用に限って使ってください
- 音質の評価: [UTMOS](https://github.com/tarepan/SpeechMOS) (SpeechMOS、MIT)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) (MIT)、[Inter](https://github.com/rsms/inter) (OFL-1.1)、[Lucide](https://lucide.dev/) (ISC)、[BudouX](https://github.com/google/budoux) 0.10.1 (Apache-2.0。`site/vendor/budoux/` に、ライセンスとともに同梱)
