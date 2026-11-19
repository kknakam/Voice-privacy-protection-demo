# Voice-privacy-protection-demo

私が研究で取り組んだ手法を用いて簡易的なデモサイトを作成するリポジトリ

その場で録音した声から敵対的音声を生成し、音声クローンへの保護効果を確かめるサイトです。
敵対的音声の生成 (Step 2) は、研究室のサーバ (Python、GPU) で行い、録音をそのサーバへ送ります。音声合成とクローンの比較 (Step 3) は、ブラウザの中で行います。

| 機能 | 内容 |
|---|---|
| 録音 | マイクから録音し、16 kHz モノラルへ変換して無音を除去、最長 8 秒に整形 |
| 敵対的音声の生成 | 録音の話者から、いちばん遠い話者を、候補 24 人から自動で選び (ターゲット話者。参照音声を試聴できる)、時間周波数マスクを最適化。反復回数は 50 / 100 / 200。「残した情報の割合」(マスクの平均値)、SNR、UTMOS (測定値 / 5.0。録音の値も添える) を表示。サーバで処理し、混み合うと順番待ち |
| クローン | 利用者が入力した英文を、録音と敵対的音声それぞれの話者埋め込みで SpeechT5 に合成させて比較 |
| 日本語のクローン | 入力した文章に、ひらがな・カタカナ・漢字が 1 字でもあれば、日本語に追加学習した SpeechT5 で合成して比較する (言語の選択欄はない。なければ英語)。モデルと辞書は同梱 |
| 研究のシナリオ | 別のタブ (URL の `#scenario` で直接開ける) で、研究が想定する場面、守る仕組み (デモの Step との対応を含む)、敵対的音声のつくり方、結果の読み方、条件と限界を、図で説明。デモの録音や結果は、タブを切り替えても残る |

話者エンコーダは生成とクローンで同じ x-vector を使うホワイトボックス条件です。生成には、研究のリポジトリ (anticloning_mask) の最適化のコードを、そのまま使います (このリポジトリには含みません)。

## 構成

```
./
  README.md
  package.json               MDX から HTML を作るのに使う (サイトの配信には使わない)
  content/scenario.mdx       「研究のシナリオ」のタブの本文 (これを編集する)
  content/demo.mdx           「デモ」のタブの本文 (これを編集する)
  tools/scenario/            MDX を HTML にする (build.mjs) と、その部品 (components.mjs: シナリオ用と共通、demo-components.mjs: デモ用)
  tools/serve.py             サイトだけを配信する簡易サーバ (site/_headers を再現。Step 2 の API は無い)
  server/                    敵対的音声を作るサーバ (Python)
    app.py                   サイトと API を 1 つで配信。順番待ち、1 件ずつ処理、録音は処理後に削除
    core.py                  最適化と測定の本体 (研究のコードを import)
    attack_cli.py            録音から敵対的音声を作るコマンド
    build_pool.py            ターゲット話者の候補群を作る (開発用)
    test_core.py             GPU を使わない部分の試験
    data/pool.npz            候補 24 人の x-vector
    env.example              環境設定の書き方 (server/env.local にコピーして書き換える。env.local は git に入れない)
  scripts/                   実行の入口のシェルスクリプト (serve.sh、run_attack.sh、build_pool.sh)
  site/                      公開する部分 (Cloudflare の出力先)
    index.html               画面 (「デモ」と「研究のシナリオ」のパネルの中身は、MDX から生成した部分。見出し画面とフッターは手書き)
    _headers                 Cloudflare 用のレスポンスヘッダ (COOP/COEP, CSP)
    css/style.css
    js/site/tabs.js          「デモ」と「研究のシナリオ」のタブ (ARIA の tabs。URL のハッシュで開く)
    js/site/scenario.js      研究のシナリオの図 3 (マスクの掛け算の模式図) を描く
    js/site/linebreak.js     Safari など、word-break: auto-phrase が使えないブラウザで、日本語を文節で折り返す (BudouX)
    js/demo/app.js           画面の制御
    js/demo/recorder.js      録音と整形
    js/demo/capture-worklet.js  録音の取り込み (AudioWorklet)
    js/demo/api.js           サーバの API (録音の送信、順番待ちと進み具合、結果の受け取り)
    js/demo/engine.js        ワーカーとの受け渡し
    js/demo/engine.worker.js 重い処理の本体
    js/demo/charts.js        スペクトログラムと波形の描画
    js/mask/dsp.js           fbank の計算 (Step 3 で、クローンの話者の特徴を取り出す)
    js/mask/vector.js        ベクトルの計算 (クローンの類似度)
    js/tts/tts.js            入力文の正規化と SpeechT5 の生成ループ
    js/tts/tts_ja.js         日本語モデルの生成ループ (文の終わりの検出、長い文の分割を含む)
    js/tts/ja_frontend.js    日本語の音素化 (OpenJTalk の WASM と、アクセント結合前の規則)
    js/tts/ja_g2p.js         フルコンテキストラベルから、音素と韻律記号の ID 列を作る
    models/xvector_grad.onnx   x-vector (16.1 MiB)。Step 3 で、クローンの話者の特徴を取り出す
    assets/constants.json    メルフィルタバンク、語彙など (攻撃の設定とターゲット話者は、サーバ側にある)
    assets/targets/          ターゲット話者の候補 24 人の参照音声 (<話者の番号>.wav。LibriTTS の発話。server/data/pool.npz と対応)
    assets/icons.svg         Lucide のアイコン
    assets/illust/           シナリオの図に使うイラスト (任意。使い方は「研究のシナリオを編集する」)
    vendor/                  ONNX Runtime Web, Inter, OpenJTalk (WASM), BudouX
    models_ja/               日本語モデルと辞書 (20 MiB ごとに分割。manifest.json に大きさと sha256)
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

日本語は、Microsoft SpeechT5 を日本語に追加学習したモデルで合成します。英語のモデルとは別に、`site/models_ja/` に同梱しています。

| 項目 | 内容 |
|---|---|
| 入力 | 自由に入力した日本語の文章 (60 文字まで) を OpenJTalk で音素と韻律記号 (アクセント句の境界、ピッチの上昇・下降など) にしたもの。ブラウザ内で、WebAssembly に変換した OpenJTalk を使う |
| 話者条件 | 英語と同じ x-vector (話者エンコーダは生成とクローンで共通) |
| モデル | encoder 178.3 MiB、decoder 103.0 MiB、vocoder 26.5 MiB (重みだけ fp16、計算は fp32)。辞書 102 MiB。20 MiB ごとに分割して置き、取得後に大きさと sha256 を確かめる |
| 学習データ | JVS コーパス (96 話者)、JSUT コーパス (basic5000)、英語を忘れないための LibriTTS の一部。計 約 33 時間、20,000 ステップ。JVS の 4 話者は学習に使わず、評価用にした |
| 初回の取得 | 約 410 MiB (モデル 約 308 MiB と、音素化の辞書 約 102 MiB) |

評価 (学習に使っていない 12 話者 x 15 文。Whisper large-v3 による文字誤り率、x-vector による話者類似度):

| | 文字誤り率 (表記) | 話者類似度 |
|---|---|---|
| 自然音声 (JSUT、同じ文) | 0.8% | — |
| 日本語モデルの合成 | 1.4% | 0.948 |

- デコーダが同じ文を 2 回読んでしまう失敗が、JVS の声で起きやすいため、デコーダの注意から文の終わりを検出して止めています (止めない場合、JVS の未知話者の文字誤り率は 16.6%)
- 長い文 (音素と韻律記号の数が 100 以上。60 文字の入力の多くが当たります) では、JVS の声に近い声で、途中の語句が抜けたり繰り返されたりすることがあり、終わりの検出では防げません。生成し直しても同じ失敗が起きるため、100 以上の文は、ポーズの位置で 60 記号以内の区間に分け、区間ごとに合成して、0.1 秒の無音を挟んでつなぎます (`site/js/tts/tts_ja.js` の `splitIds` と `joinWaves`。区間ごとに合成する処理は `site/js/demo/engine.worker.js`)。設定は評価の前に固定し、設定の選択に使っていない新しい長文 24 文を、話者 8 人、本人と保護後の声で合成して、読みの文字誤り率が 0.2 を超えた件を数えました。JVS の話者 4 人で 31/192 (16.1%) から 1/192 (0.5%)、JVS 以外の話者 4 人で 0/192 のまま (悪化した件はなし) でした。つなぎ目の自然さ (間、声の高さの段差) は、自動では測っていません
- 自由入力の音素化は、OpenJTalk の素の版です。学習時の音素化 (Sudachi による読みの補正などを含む) とは、約 4 分の 1 の文で記号列が違います。聞き取りへの影響は小さいと確認していますが、アクセントが不自然になることがあります
- 評価は少数の話者と文で行ったものです。アクセントの自然さは、文字誤り率では測れていません

## 実装の考え方

処理は、2 か所に分かれています。

| 場所 | 処理 |
|---|---|
| サーバ (Python、GPU) | Step 2: 録音の x-vector を求め、候補 24 人のうち、いちばん遠い話者をターゲットにする。研究の `run_mask_attack` (Adam、勾配のクリッピング、Beta 損失、音響損失) でマスクを最適化する。残した情報の割合 (マスクの平均値)、SNR、UTMOS を測る |
| ブラウザ | Step 3: 録音と敵対的音声の x-vector (サーバが返す) を話者条件にして、SpeechT5 で合成する。合成した声の x-vector を、ONNX Runtime Web で取り出して、本人・ターゲットとの類似度を測る |

サーバの流れ: `POST /api/jobs` で WAV を受け取ると、順番待ちの列に入れます。GPU は 1 件ずつ使い、`scripts/run_attack.sh` を、別のプロセスで呼びます。録音は、処理が終わるとすぐ、サーバのディスクから消し、結果 (音声、統計、x-vector、スペクトログラム) は、取りに来るまでメモリにだけ置きます (取ると消え、取りに来なければ 10 分で捨てます)。

- **入力の確認:** 16 kHz・モノラル・16 bit の WAV だけを受け付けます (ブラウザは、この形で送ります)。大きさは 1 MiB まで、長さは 8 秒までです。反復回数は 50 / 100 / 200 だけです。シェルに渡す引数は、サーバが決めた値だけで、利用者の文字列は渡しません
- **順番待ち:** 同じ接続元から、同時に出せるのは 1 件です。順番待ちが 5 件を超えると、受け付けません (環境変数 `MAX_QUEUE` で変えられます)。1 件の上限は 300 秒です
- **ターゲットの候補:** LibriTTS の 101 人の x-vector の平均から、互いに遠い 24 人を選びました (最遠点サンプリング。話者どうしの類似度の平均は 0.904)。`scripts/build_pool.sh` で作り直せます

## 研究のシナリオ (タブ)

- **タブ:** `site/js/site/tabs.js` が、「デモ」と「研究のシナリオ」のパネルを `hidden` で切り替えます。パネルは消さないので、録音、生成結果、入力した文章は、タブを往復しても残ります。タブごとにスクロール位置を覚え、再生中の音声は、タブを離れると一時停止します。処理中は、「デモ」のタブに印が出ます
- **URL:** `#scenario` で直接開けます。`#step-clone` のような、デモ内の要素を指すハッシュでも、タブを切り替えてその位置へ移動します。戻る・進むにも対応します。`app.js` とは独立なので、エンジンの読み込みが失敗しても、タブは使えます
- **図:** 流れの図は HTML と CSS だけで描きます (1040px 以上で横並び、それより狭いと縦並び)。対策の図は、「加工しない」と「加工する」で共通の手順 (本人の声、公開・入手、声の特徴を数値にする) を 1 枚のカードにして、上下 2 段のカードの真ん中に置きます。そこから出る 2 本の矢印は、共通のカードの高さの 1/4 と 3/4 から伸びます。違いのある手順 (公開前の加工、合成された声) だけを、上下 2 段に分けます。カードは、すべて高さ 208px です。カードの中の文字は、中央に揃えます。これらの寸法は、`site/css/style.css` の「分岐の図の寸法」にまとめてあります (本文を大きく増やして、カードが 208px を超えると、矢印の高さがずれます)。「加工しない」の段は、本人の声から公開・入手まで、長い 1 本の矢印で結びます。合成された声のカードには、目的が達成されたか (丸 = 守れる、ばつ = 守れない) を、アイコンと文字で示します。デモの Step への移動ボタンは、該当するカードの下に付きます。最適化の手順の図は、最後の手順の下から最初の手順の下へ戻る矢印で、くり返しを示します。結果の読み方の図は、人間と機械の受け取り方の違いを表す SVG の模式図です。図 3 の格子は `site/js/site/scenario.js` が描きます。CSP (`style-src 'self'`) に合わせて、インラインの style と script は使いません
- **色:** 紫 = 本人、青緑 = 保護、橙 = 攻撃者 (クローン)。各カードに、役割を文字でも書いています
- **内容:** タブは、研究の想定場面と仕組みの解説に絞っています。検証の数値は載せず、下の「検証結果」に書いています
- **アイコン:** [Lucide](https://lucide.dev/) 1.74.0 の SVG を、`site/assets/icons.svg` に取り込んでいます

## 本文を編集する (MDX)

2 つのタブの本文は、Markdown と部品 (JSX) で書きます。編集したら、次を実行して `site/index.html` に反映します。Node.js 22 以上が要ります。

| タブ | 本文 | 部品の説明 |
|---|---|---|
| 研究のシナリオ | `content/scenario.mdx` | `tools/scenario/components.mjs` |
| デモ | `content/demo.mdx` | `tools/scenario/demo-components.mjs` (シナリオの部品の `Title`、`Lead`、`Hint` も使う) |

```bash
npm install                  # 初回だけ
npm run build                # 2 つの MDX から HTML を生成し、site/index.html の印の間を書き換える
npm run check                # 書き換えずに、index.html が MDX と一致しているかを確かめる (食い違うと終了コード 1)
npm run build:demo           # デモだけ (build:scenario は、シナリオだけ)
```

- `site/index.html` の `BEGIN scenario-content` / `BEGIN demo-content` と、それぞれの `END` の間は自動生成です。手で直しても、次の生成で消えます
- 書き間違い (存在しない図への参照、足りない項目、同じ id など) は、`エラー: …` の 1 行で、MDX の行番号とともに知らせます
- 配信されるのは、生成した HTML だけです。MDX も Node.js も、ブラウザには届きません
- **シナリオの部品:** `Block` (節)、`Figure` (図。番号は自動)、`Ref` (「図 N」への参照)、`Flow` と `Step` (流れの図)、`Branching` / `Alts` / `Case` (共通の手順と、2 本に分かれる手順を持つ流れ)、`Flow` の `loop` (くり返しの矢印)、`Step` の `result` / `resultLabel` (丸とばつ)、`Perception` (人間と機械の模式図。文字だけ変えられる)、`Hint` / `Notice` / `Summary` (補足、注意、要点)
- **デモの部品:** `DemoStep` (ステップ)、`Button` / `Actions` / `Timer` (ボタン)、`Segmented` / `Option` (選択肢)、`TextField` (文章の入力欄)、`AudioRow` / `AudioGrid` (再生器)、`Heat` / `HeatGrid` / `HeatScale` (スペクトログラム)、`CloneCard` / `CloneGrid` (クローンの結果)、`Progress` / `Alert` / `Result` / `Stats` / `Slot` / `Waveform` / `EnvNotice` (JS が、あとから中身を入れる場所)
- **デモの文章は、自由に書き換えられますが、id は変えられません。** `record-start` のような id は、`site/js/demo/app.js` が、部品を探す名前です。ビルドは、JS が探す id が HTML に無いと、`エラー: JS が探す id と、HTML が合っていません` で止まります。ボタンや入力欄を増やす場合は、`app.js` の変更も要ります
- 画面を動かす文 (進み具合、エラー、結果の説明など) は、MDX ではなく、`app.js` にあります。MDX で書き換えられるのは、画面の最初の状態の文章と、構造です

### イラストを使う

カードのアイコンの代わりに、絵を出せます。

1. 絵のファイルを `site/assets/illust/` に置きます (SVG を推奨。PNG・WebP・JPEG も可。1 枚 300 KiB 以下)
2. `content/scenario.mdx` の `<Step>` に、`image="ファイル名" alt="絵の説明"` を足します。例: `<Step role="本人" icon="mic" color="purple" title="本人の声" image="speaker.svg" alt="マイクに向かって話す人">`。`alt` は必須です (飾りだけの絵なら `alt=""`)。`icon` は、絵を外したときの代わりなので、残します
3. `npm run build:scenario` を実行します。ファイルが無い、形式が違う、大きすぎる、`alt` が無い場合は、エラーで知らせます

絵を作るときの決まり:

| 項目 | 内容 |
|---|---|
| 縦横比 | 4:3 (例: 240 x 180)。枠の中に、縦横比を保って収まる |
| 背景 | 透明にする。枠の背景に、役割の色の薄い色が付く (紫 `#f3eafc`、青緑 `#e4f5f4`、橙 `#fdeee2`、灰 `#f1f0ee`) |
| 色 | 役割の色に合わせると、図全体がそろう。紫 = 本人 (`#d6b6f6` / `#391c57`)、青緑 = 保護 (`#cfeeec` / `#125a58`)、橙 = 攻撃者 (`#fbe0cc` / `#793400`)、灰 = 中立 (`#ebeae8` / `#31302e`) |
| 文字 | 絵の中に入れない。説明は、カードの文章 (HTML) で書く。絵の中の文字は、編集も読み上げもできない |
| SVG | `<script>`、外部ファイルへの参照、埋め込みフォントは使わない。画像として読み込むので、動かない |
| 素材 | 自作か、再利用が許されているものにする。他人の素材は、ライセンスの表示を、下の「クレジット」に足す |

## 検証結果

以前の版 (最適化をブラウザの WebAssembly で行い、ターゲットは男女から選ぶ形) で測った結果です。Chromium の疑似マイクに LibriTTS の音声を入力し、画面操作で全工程を通しました
(5 話者、各 8 秒、100 反復、利用者入力を想定した 5 種類の英文)。

| クローンの元 | 本人との類似度 | ターゲットとの類似度 | FSR | WER |
|---|---|---|---|---|
| 録音 (保護なし) | 0.962 | 0.880 | 0% | 0.062 |
| 敵対的音声 (保護あり) | 0.878 | 0.955 | 100% | 0.077 |

- 画面に表示された類似度は、保存した音声を PyTorch の x-vector で測り直した値と一致しました
- WER の差の主因は、Whisper が "twenty-five dollars" を "$25" と書き起こすなどの表記揺れです
- **サーバ版 (現行) の再測定は、未実施です。** 動作の確認だけ行いました: 録音 (疑似マイク) から、サーバでの生成、結果の表示、日本語のクローンの判定まで、画面の操作で通しました。録音 1 件 (約 6 秒) の生成は、反復 100 回で、サーバ内の処理が約 6 秒 (GPU)、モデルの読み込みと音質の測定を含めた全体で約 16 秒です
- 1 件の録音で測った例 (反復 100 回): 残した情報の割合 (マスクの平均値) 約 91%、SNR 約 6.5 dB、UTMOS 4.17 → 3.67。ターゲットとの類似度が、0.874 から 0.974 に上がります。なお、元の音声のエネルギーで重みを付けると、残る分は 45% ほどです (エネルギーの大きい成分が、先に削られるため)。画面には、マスクの平均値を出します

実際のマイク録音、Firefox / Safari、一般的な PC での速度は未検証です。

## ローカルで動かす

マイクは `localhost` か HTTPS でのみ使えます。COOP/COEP が付かないとシングルスレッドになるので、`site/_headers` を再現するサーバを使います。

**Step 2 まで使う場合 (サーバ):** Python (PyTorch、speechbrain、soundfile)、GPU、研究のリポジトリ、重みが要ります。

```bash
cp server/env.example server/env.local      # 研究のリポジトリ、Python、重みの場所を書く。env.local は git に入れない
scripts/check_env.sh --smoke                 # 環境の点検 (重み、GPU、候補話者など)。--smoke で、生成を 1 回、実際に通す
scripts/serve.sh 8000                        # http://localhost:8000/ (サイトと API。前面で動かす)
scripts/start.sh 8000                        # 同じものを、バックグラウンドで起動 (止める: scripts/stop.sh、状態: scripts/status.sh)
python -m unittest server.test_core -v       # GPU を使わない部分の試験
scripts/run_attack.sh --input 録音.wav --out-dir 出力先 --steps 100   # 生成だけを、コマンドで試す
```

マイクは、`http://localhost:ポート番号/` か `https://` の URL でしか使えません。サーバの IP アドレスや、ホスト名の `http://` の URL で開くと、ブラウザがマイクを止めるため、Step 1 の録音は押せず、画面に理由が出ます。リモートのサーバを使う場合は、SSH や VS Code のポート転送で、手元の `localhost` に出すか、下の「公開する」の HTTPS を使います。

重みは、実行時にネットへ取りに行きません (`HF_HUB_OFFLINE=1`)。x-vector (`speechbrain/spkrec-xvect-voxceleb`) は `HF_HOME` の下に、UTMOS (`tarepan/SpeechMOS` v1.2.0 と `utmos22_strong_step7459_v1.pt`) は `TORCH_HUB_DIR` の下に、あらかじめ置きます。

**サイトだけ見る場合 (Step 2 は使えません):**

```bash
python tools/serve.py                        # http://localhost:8000/ (リポジトリの直下で実行)
```

このサーバは、`site/_headers` に `Cache-Control` の指定がないファイルに `no-cache` を付け、編集した内容が、ブラウザの古い保存分に隠れないようにします。以前のサーバで保存された古いファイルが残っていて、表示が更新されないときは、DevTools の Network で「Disable cache」にして再読み込みするか、ブラウザのキャッシュを消去してください。

## 公開する (研究室のサーバ + Cloudflare Tunnel)

Step 2 は Python と GPU が要るので、静的な配信だけ (Cloudflare Pages など) では動きません。サイト全体を、研究室のサーバから `scripts/serve.sh` で配信します。

1. サーバに、上の「ローカルで動かす」と同じ環境を作ります。`scripts/check_env.sh --smoke` で、足りないものを確かめます
2. 常駐させます。systemd が使える環境では、`deploy/voice-demo.service` の雛形の `User` とパスを書き換えて、`/etc/systemd/system/` に置き、`systemctl enable --now voice-demo` します。使えない環境 (コンテナなど) では、`scripts/start.sh 8000` を使います (再起動のたびに、起動し直す必要があります)。サーバは、既定で `127.0.0.1` だけで待ち受けます
3. HTTPS で公開します (マイクは、HTTPS でしか使えません)。Cloudflare Tunnel (`cloudflared`) を使うと、サーバから外向きに接続するだけで、ポートを開けずに、HTTPS で公開できます。
   - **試す (アカウント不要):** `scripts/tunnel.sh quick 8000` で、一時的な URL (`https://....trycloudflare.com`) が出ます。URL を知っている人は、誰でも、この処理 (GPU) を使えます。確認が終わったら、Ctrl-C で閉じてください
   - **本番 (名前つき):** Cloudflare に登録したドメインが要ります。`deploy/cloudflared-config.example.yml` の手順で、トンネルを作り、`scripts/tunnel.sh run <トンネル名>` で起動します
   - `cloudflared` は、環境変数 `CLOUDFLARED`、`server/work/bin/cloudflared`、`PATH` の順に探します (公式の配布: https://github.com/cloudflare/cloudflared/releases)
4. 前段 (Tunnel) から来る接続の元の IP アドレスは、`CF-Connecting-IP` から読みます (同じ接続元からの同時の処理を 1 件に絞るため)。サーバを、`127.0.0.1` 以外で待ち受けさせる場合は、この読み取りを使わないよう、前段を必ず通してください

録音がサーバに届くので、研究室の規定 (同意の取り方、保存、ログ) に合わせて、フッターの「プライバシー」の文面を確認してください。

`site/models_ja/` の分割ファイルは、1 つが 20 MiB です (Cloudflare Pages の 1 ファイル 25 MiB の上限に合わせた名残です)。

## 素材について

`site/models/xvector_grad.onnx` と `site/assets/constants.json` は、研究用のコードから生成したものです。生成用のコードはこのリポジトリに含みません。

攻撃の設定 (学習率、損失の重みなど) は、`server/core.py` の `ATTACK` にあります。Beta の β だけは、Beta 探索の採用値である 1.0 に揃えています (研究の設定ファイルは 0.5)。

## 制限

- 英語のモデル (SpeechT5) は英語専用です。録音は他の言語でも構いません。読み上げる言語は、入力した文章から決めます (日本語の文字が 1 字でもあれば日本語)。日本語のモデルが無い環境では、日本語は入力できても、読み上げられません
- 初回に、英語で約 189 MiB、日本語で約 410 MiB (モデル約 308 MiB と辞書約 102 MiB) を取得します。日本語は、ブラウザのメモリも多く使います (開発環境の実測で 1.2〜1.5 GB 程度)。PC のブラウザを推奨します
- 日本語のモデルは、JVS の声に近い x-vector で、短い文を、同じ文の繰り返しにしてしまうことがあります。終わりの検出で止めていますが、注意が崩れて末尾に届かない場合には効きません。長い文では、途中の語句の抜けや繰り返しが起きやすいため、ポーズの位置で区間に分けて合成しています (上の「日本語のクローン」)。ポーズのない長い数字の列 (16 桁など) は、分けられないので、読み誤ることがあります
- SpeechT5 は推論時も dropout が有効なので、同じ入力でも合成結果と類似度は毎回変わります
- 敵対的音声の生成は、サーバで 1 件ずつ処理するため、混み合うと順番待ちになります。サーバに GPU と、研究のリポジトリ、重みが要ります
- 効果はホワイトボックス条件のものです。他の音声合成モデルへの保護効果は示していません
- 日本語のクローンでの保護効果は、系統立てた評価をしていません (評価したのは、日本語モデルの合成品質です)
- 日本語の折り返しは、`word-break: auto-phrase` で、文節の途中で切らないようにしています。この指定は、Chrome と Edge の 119 以降で効きます。効かないブラウザ (Safari の安定版、Firefox など) では、`site/js/site/linebreak.js` が、BudouX で文を文節に分けて `<wbr>` を入れ、同じ見え方にします (Chrome では、何もしません)。Safari 実機での確認は、未実施です

## クレジット

- 話者エンコーダ: [SpeechBrain spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) (Apache-2.0)
- 音声合成: [Microsoft SpeechT5](https://huggingface.co/microsoft/speecht5_tts) / [HiFi-GAN](https://huggingface.co/microsoft/speecht5_hifigan) (MIT)、ONNX 変換は [Xenova/speecht5_tts](https://huggingface.co/Xenova/speecht5_tts)
- ターゲット話者の埋め込みと参照音声: LibriTTS (CC BY 4.0)。参照音声は話者 1066（女性）と 7553（男性）の発話を 16 kHz モノラルに変換し、音量を調整したもの
- 日本語の音素化: [Open JTalk](https://open-jtalk.sourceforge.net/) と NAIST Japanese Dictionary (いずれも修正 BSD)。WebAssembly に変換して使用。ライセンスは `vendor/openjtalk/` に同梱
- 日本語モデルの学習データ: [JVS コーパス](https://arxiv.org/abs/1908.06248) と [JSUT コーパス](https://arxiv.org/abs/1711.00354) (東京大学)、LibriTTS の一部 (CC BY 4.0)。**JVS と JSUT の音声の利用は、学術機関の研究、非営利の研究 (企業内を含む)、個人利用に限られ、再配布は認められていません。** このデモの日本語モデルは、その条件に従い、非営利の研究・個人利用に限って使ってください
- 音質の評価: [UTMOS](https://github.com/tarepan/SpeechMOS) (SpeechMOS、MIT)
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (MIT)、[Inter](https://github.com/rsms/inter) (OFL-1.1)、[Lucide](https://lucide.dev/) (ISC)、[BudouX](https://github.com/google/budoux) 0.10.1 (Apache-2.0。Safari など、`word-break: auto-phrase` が使えないブラウザで、日本語を文節の途中で折り返さないために使う。`site/vendor/budoux/` に、ライセンスとともに同梱)
