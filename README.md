# Voice-privacy-protection-demo

私が研究で取り組んだ手法を用いて簡易的なデモサイトを作成するリポジトリ

その場で録音した声から敵対的音声を生成し、音声クローンへの保護効果を確かめるサイトです。
敵対的音声の生成 (Step 2) と、音声合成によるクローンの比較 (Step 3) は、どちらも研究室のサーバ (Python、GPU) で行い、録音と、入力した文章を、そのサーバへ送ります。ブラウザは、録音、画面、再生を受け持ちます。

| 機能 | 内容 |
|---|---|
| 録音 | マイクから録音し、16 kHz モノラルへ変換して無音を除去、最長 8 秒に整形 |
| 敵対的音声の生成 | 録音の話者から、いちばん遠い話者を、候補 24 人から自動で選び (ターゲット話者。参照音声を試聴できる)、時間周波数マスクを最適化。反復回数は 50 / 100 / 200。「残した情報の割合」(マスクの平均値)、SNR、UTMOS (測定値 / 5.0。録音の値も添える) を表示。サーバで処理し、混み合うと順番待ち |
| クローン | 利用者が入力した文章を、録音と敵対的音声それぞれの話者埋め込みで SpeechT5 に合成させて比較 (サーバで合成。英語は PyTorch、日本語は ONNX Runtime) |
| 日本語のクローン | 入力した文章に、ひらがな・カタカナ・漢字が 1 字でもあれば、日本語に追加学習した SpeechT5 で合成して比較する (言語の選択欄はない。なければ英語)。モデルは `server/models_ja/` にあり、サーバが読み込む |
| 研究のシナリオ | 別のタブ (URL の `#scenario` で直接開ける) で、研究が想定する場面、守る仕組み (デモの Step との対応を含む)、敵対的音声のつくり方、結果の読み方、条件と限界を、図で説明。デモの録音や結果は、タブを切り替えても残る |

話者エンコーダは生成とクローンで同じ x-vector を使うホワイトボックス条件です。生成には、研究のリポジトリ (anticloning_mask) の最適化のコードを、そのまま使います (このリポジトリには含みません)。

## 構成

```
./
  README.md
  package.json               MDX から HTML を作るのに使う (サイトの配信には使わない)
  requirements.txt           サーバ (Step 2) の Python の依存パッケージ (版を固定)
  content/scenario.mdx       「研究のシナリオ」のタブの本文 (これを編集する)
  content/demo.mdx           「デモ」のタブの本文 (これを編集する)
  tools/scenario/            MDX を HTML にする (build.mjs) と、その部品 (components.mjs: シナリオ用と共通、demo-components.mjs: デモ用)
  tools/serve.py             サイトだけを配信する簡易サーバ (site/_headers を再現。Step 2 の API は無い)
  server/                    敵対的音声を作るサーバ (Python)
    app.py                   サイトと API を 1 つで配信。敵対的音声の生成と、クローンの合成は、別の順番待ちで 1 件ずつ処理。録音と文章は処理後に削除
    core.py                  最適化と測定の本体 (研究のコードを import)
    tts.py                   音声合成 (Step 3) の本体: 英語の文章の正規化、日本語の音素化・分割・生成、x-vector による類似度。モデルは、起動時に 1 回だけ読み込む
    attack_cli.py            録音から敵対的音声を作るコマンド
    build_pool.py            ターゲット話者の候補群を作る (開発用)
    test_core.py             敵対的音声の生成のうち、GPU を使わない部分の試験
    test_tts.py              音声合成のうち、モデルを使わない部分の試験 (英語の正規化、日本語の音素化と分割)
    setup_server.py          研究室のサーバへの構築を自動で行う本体 (scripts/setup_server.sh から呼ぶ。標準ライブラリだけ)
    test_setup.py            構築の自動化の、判断の部分の試験 (ドライバに合う PyTorch の選択、venv が使えない環境への切り替えなど)
    check_api.py             起動中のサーバの、クローンの合成の API の通し試験 (モデルを実際に使う)
    data/pool.npz            候補 24 人の x-vector
    data/tts.json            音声合成の語彙と記号表 (モデルの学習時のものと同じ。変更しない)
    models_ja/               日本語の音声合成のモデル (ONNX。20 MiB ごとに分割。manifest.json に大きさと sha256)
    testdata/                試験の正解データ (ブラウザで動かしていた実装の結果など)
    env.example              環境設定の書き方 (server/env.local にコピーして書き換える。env.local は git に入れない)
    weights/*.sha256         重み (x-vector、英語の SpeechT5、UTMOS) の SHA-256 の一覧 (scripts/copy_weights.sh が照合に使う)
  scripts/                   実行の入口のシェルスクリプト (setup_server.sh (研究室のサーバへの構築を自動で行う)、serve.sh、start.sh / stop.sh / status.sh、run_attack.sh、check_env.sh、copy_weights.sh、tunnel.sh、build_pool.sh)
  deploy/                    研究室のサーバへ置くための雛形 (systemd、Cloudflare Tunnel、nginx)
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
    js/demo/api.js           サーバの API (録音の送信、クローンの依頼、入力中の案内、順番待ちと進み具合、結果の受け取り)
    js/demo/charts.js        スペクトログラムと波形の描画
    assets/targets/          ターゲット話者の候補 24 人の参照音声 (<話者の番号>.wav。LibriTTS の発話。server/data/pool.npz と対応)
    assets/icons.svg         Lucide のアイコン
    assets/illust/           シナリオの図に使うイラスト (任意。使い方は「研究のシナリオを編集する」)
    vendor/                  Inter、Lucide、BudouX
```

## 英語のクローン

英語は、Microsoft SpeechT5 と HiFi-GAN (PyTorch、`transformers`) で、サーバが合成します (`server/tts.py`)。GPU があれば GPU で動かします (環境変数 `TTS_DEVICE`)。CPU でも動きますが、94 文字の文で 1 本あたり約 10 秒かかります (GPU では約 2.7 秒、GPU のメモリ 約 710 MiB)。

- **文章の正規化:** 語彙が英語の文字と一部の記号だけなので、数字を英単語に展開し (`12` は `twelve`、`3rd` は `third`、`$5` は `five dollars`、`50%` は `fifty percent`)、語彙にない文字を除きます。英語以外の文字 (ギリシャ文字、キリル文字など) があれば、英語で入力するよう案内します。200 文字までです。この規則は、以前ブラウザで動かしていた実装と同じで、66 件の入力で、結果が一致することを確かめています (`server/testdata/normalize_en.json`、`server/test_tts.py`)
- **話者条件:** 録音と敵対的音声の x-vector (サーバが Step 2 の結果として返したもの)
- **上限:** 止まらない場合に備えて、入力長の「10 倍」と「6 倍 + 100」の小さいほうのステップ数で止めます

## 日本語のクローン

日本語は、Microsoft SpeechT5 を日本語に追加学習したモデルで、サーバが合成します。モデルは、リポジトリの `server/models_ja/` にあります (公開用の `site/` には含めません)。

| 項目 | 内容 |
|---|---|
| 入力 | 自由に入力した日本語の文章 (60 文字まで) を、サーバの pyopenjtalk-plus (Open JTalk) で、音素と韻律記号 (アクセント句の境界、ピッチの上昇・下降など) にしたもの |
| 話者条件 | 英語と同じ x-vector (話者エンコーダは生成とクローンで共通) |
| モデル | encoder 178.3 MiB、decoder 103.0 MiB、vocoder 26.5 MiB (重みだけ fp16、計算は fp32。ONNX Runtime、CPU)。20 MiB ごとに分割して置き、読み込むときに、大きさと sha256 を確かめる |
| 学習データ | JVS コーパス (96 話者)、JSUT コーパス (basic5000)、英語を忘れないための LibriTTS の一部。計 約 33 時間、20,000 ステップ。JVS の 4 話者は学習に使わず、評価用にした |
| 読み込みと速度 | サーバの起動時に 1 回だけ読み込む (英語のモデルと合わせて約 20 秒)。1 回の合成 (2 本) は、短い文で約 4 秒、長い文 (入力長 100 以上) で約 7〜9 秒 (CPU 4 スレッド、他の処理と並行した状態での測定)。以前のブラウザ版は、11〜19 秒でした |

評価 (学習に使っていない 12 話者 x 15 文。Whisper large-v3 による文字誤り率、x-vector による話者類似度):

| | 文字誤り率 (表記) | 話者類似度 |
|---|---|---|
| 自然音声 (JSUT、同じ文) | 0.8% | — |
| 日本語モデルの合成 | 1.4% | 0.948 |

- デコーダが同じ文を 2 回読んでしまう失敗が、JVS の声で起きやすいため、デコーダの注意から文の終わりを検出して止めています (止めない場合、JVS の未知話者の文字誤り率は 16.6%)
- 長い文 (音素と韻律記号の数が 100 以上。60 文字の入力の多くが当たります) では、JVS の声に近い声で、途中の語句が抜けたり繰り返されたりすることがあり、終わりの検出では防げません。生成し直しても同じ失敗が起きるため、100 以上の文は、ポーズの位置で 60 記号以内の区間に分け、区間ごとに合成して、0.1 秒の無音を挟んでつなぎます (`server/tts.py` の `split_ids` と `JaSynth.synthesize`)。設定は評価の前に固定し、設定の選択に使っていない新しい長文 24 文を、話者 8 人、本人と保護後の声で合成して、読みの文字誤り率が 0.2 を超えた件を数えました。JVS の話者 4 人で 31/192 (16.1%) から 1/192 (0.5%)、JVS 以外の話者 4 人で 0/192 のまま (悪化した件はなし) でした。つなぎ目の自然さ (間、声の高さの段差) は、自動では測っていません
- 自由入力の音素化は、OpenJTalk の素の版です。学習時の音素化 (Sudachi による読みの補正などを含む) とは、約 4 分の 1 の文で記号列が違います。聞き取りへの影響は小さいと確認していますが、アクセントが不自然になることがあります
- 評価は少数の話者と文で行ったものです。アクセントの自然さは、文字誤り率では測れていません

## 実装の考え方

処理は、サーバとブラウザに分かれています。

| 場所 | 処理 |
|---|---|
| サーバ (Python、GPU) | Step 2: 録音の x-vector を求め、候補 24 人のうち、いちばん遠い話者をターゲットにする。研究の `run_mask_attack` (Adam、勾配のクリッピング、Beta 損失、音響損失) でマスクを最適化する。残した情報の割合 (マスクの平均値)、SNR、UTMOS を測る。<br>Step 3: 文章を、英語は正規化し、日本語は音素化して、録音と敵対的音声の x-vector で、SpeechT5 で 2 本合成する。合成した声の x-vector (SpeechBrain) を取り出して、本人・ターゲットとの類似度を測る |
| ブラウザ | 録音と整形、画面の制御、スペクトログラムと波形の描画、音声の再生。サーバへは、録音 (Step 2)、文章と x-vector (Step 3) を送り、結果を受け取ります |

サーバの流れ: `POST /api/jobs` で WAV を受け取ると、順番待ちの列に入れます。GPU は 1 件ずつ使い、`scripts/run_attack.sh` を、別のプロセスで呼びます。録音は、処理が終わるとすぐ、サーバのディスクから消し、結果 (音声、統計、x-vector、スペクトログラム) は、取りに来るまでメモリにだけ置きます (取ると消え、取りに来なければ 10 分で捨てます)。

Step 3 の流れ: `POST /api/clones` で、文章と、Step 2 の結果の x-vector (本人、保護後、ターゲット) を受け取ります。合成は、サーバのプロセスの中で 1 件ずつ行い (モデルは、起動時に 1 回だけ読み込みます)、敵対的音声の生成とは別の順番待ちで、同時に動けます。文章は、処理が終わるとすぐに捨て、サーバの記録には、言語と文字数と所要時間だけを残します (文章そのものは残しません。予期しない例外の記録も、型名だけです)。入力中の案内 (正規化した文、エラー) は `POST /api/normalize` で返します (モデルは使いません)。

- **入力の確認:** 録音は、16 kHz・モノラル・16 bit の WAV だけを受け付けます (ブラウザは、この形で送ります)。大きさは 1 MiB まで、長さは 8 秒までです。反復回数は 50 / 100 / 200 だけです。文章は、日本語 60 文字、英語 200 文字までで、依頼の JSON は 32 KiB まで、x-vector は、長さ 512 の有限な数列だけを受け付けます。シェルに渡す引数は、サーバが決めた値だけで、利用者の文字列は渡しません
- **順番待ち:** 同じ接続元から、同時に出せるのは 1 件です (Step 2 と Step 3 をまたいで)。順番待ちは、それぞれ 5 件を超えると、受け付けません (環境変数 `MAX_QUEUE` で変えられます)。敵対的音声の生成は 1 件 300 秒、クローンの合成は 2 本で 180 秒が上限です
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
- **Step 2 (敵対的音声) の、サーバ版での FSR と WER の再測定は、未実施です。** 動作の確認だけ行いました: 録音 (疑似マイク) から、サーバでの生成、結果の表示、クローンの判定まで、画面の操作で通しました。録音 1 件 (約 6 秒) の生成は、反復 100 回で、サーバ内の処理が約 6 秒 (GPU)、モデルの読み込みと音質の測定を含めた全体で約 16 秒です
- 1 件の録音で測った例 (反復 100 回): 残した情報の割合 (マスクの平均値) 約 91%、SNR 約 6.5 dB、UTMOS 4.17 → 3.67。ターゲットとの類似度が、0.874 から 0.974 に上がります。なお、元の音声のエネルギーで重みを付けると、残る分は 45% ほどです (エネルギーの大きい成分が、先に削られるため)。画面には、マスクの平均値を出します

### サーバ版の音声合成の評価 (Step 3)

音声合成をサーバに移したあとの評価です。話者の x-vector (本人の録音、保護後) で 2 本ずつ合成し、ブラウザ版と同じ指標 (Whisper large-v3 の誤り率、x-vector の類似度) で測りました。

| 言語と文 | 指標 | サーバ版 | 基準 |
|---|---|---|---|
| 日本語、JSUT の 15 文 × 6 話者 | 読みの文字誤り率が 0.2 を超えた件 | JVS の声に近い 3 話者で 1/90、それ以外の 3 話者で 0/90 | 同じ話者・同じ文のブラウザ版の評価で、2/90 と 0/90 |
| 日本語、入力長 100 以上の新しい長文 12 文 × 6 話者 | 同上 | 0/72 と 0/72 (平均の読みの誤り 0.36% と 0.00%) | 同じ話者・同じ文の分割ありの評価で、0/72 と 0/72 |
| 英語、8 文 × 12 話者 | 単語誤り率 (WER) | 保護なし 0.029、保護あり 0.014 | 以前のブラウザ版 (5 話者、5 文): 0.062 と 0.077 |
| 英語 | x-vector の類似度 (本人 / ターゲット) | 保護なし 0.954 / 0.815、保護あり 0.849 / 0.933 | 以前のブラウザ版: 0.962 / 0.880 と 0.878 / 0.955 |

- 英語の比較は、話者、文、ターゲットの選び方、x-vector の測り方が、以前とは違うので、絶対値の比較ではなく、同じ水準かを見るためのものです。WER は、数字を英単語に展開してから比べています
- 日本語の話者 (JVS) の x-vector で英語を読ませると、まれに文を繰り返します (96 組のうち 2 件)。英語には、日本語のような、文の終わりの検出がありません
- 実ブラウザ (ヘッドレス Chromium) で、録音から日本語・長文・英語のクローンまでを通しました。日本語のクローンは約 4〜5 秒で、以前のブラウザ版の 11〜19 秒より速く、ブラウザは、モデル、辞書、WASM を取得しません (通信の記録で確認)
- サーバのプロセスは、モデルを常駐させるため、メモリ (RAM) を約 3.0 GiB、GPU のメモリを約 1.4 GiB 使います (起動から数分後の実測)

実際のマイク録音、Firefox / Safari、一般的な PC での速度は未検証です。

## ローカルで動かす

マイクは `localhost` か HTTPS でのみ使えます。COOP/COEP が付かないとシングルスレッドになるので、`site/_headers` を再現するサーバを使います。

**Step 2 まで使う場合 (サーバ):** Python (`requirements.txt`)、GPU、研究のリポジトリ、重みが要ります。

```bash
pip install -r requirements.txt              # PyTorch、speechbrain など (版は固定。PyTorch の CUDA の版は、requirements.txt の説明を参照)
cp server/env.example server/env.local      # 研究のリポジトリ、Python、重みの場所を書く。env.local は git に入れない
scripts/copy_weights.sh --from-hf 元の HF_HOME --from-hub 元の TORCH_HUB_DIR   # 重み (x-vector、UTMOS) を置く。SHA-256 で照合する (詳しくは「公開する」)
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

## 公開する (研究室のサーバ)

Step 2 は Python と GPU が要るので、静的な配信だけ (Cloudflare Pages など) では動きません。サイト全体を、研究室のサーバから `scripts/serve.sh` で配信します。Step 3 (音声合成) は、ブラウザで動くので、サーバは使いません。

### サーバに要るもの

| もの | 内容 |
|---|---|
| OS と GPU | Linux、NVIDIA の GPU。CPU だけでは、非常に遅くなります (`scripts/run_attack.sh --device cpu`) |
| 1 件の生成の負荷 | 敵対的音声 (5.7 秒の録音、100 反復): 約 16 秒、GPU のメモリ 約 1.4 GiB、メモリ 約 2.7 GiB。クローンの合成: 英語は GPU で、2 本で約 2 秒。日本語は CPU で、短い文 約 4 秒、長い文 約 7〜9 秒。サーバのプロセスは、モデルを常駐させるので、メモリ 約 3.0 GiB と GPU のメモリ 約 1.4 GiB を、常に使います。2 つの処理は、それぞれ 1 件ずつです。録音が 8 秒 (上限) の場合の値と、他の GPU での動作は、未確認です |
| Python | 3.10 で動作確認。依存は `requirements.txt` です。PyTorch の CUDA の版は、GPU とドライバに合わせます |
| 研究のリポジトリ | `anticloning_mask` のコード (約 0.5 MB)。`ANTICLONING_MASK_ROOT` で指します。変更しません |
| 重み | x-vector 約 32 MB、英語の SpeechT5 とボコーダ 約 1.2 GB (使うのは約 640 MB)、UTMOS 約 411 MB。実行時にネットから取らないので、あらかじめ置きます (手順 5)。日本語の音声合成のモデル (約 308 MiB) は、リポジトリの `server/models_ja/` にあります |
| HTTPS | マイクは HTTPS か `localhost` でしか使えません。Cloudflare Tunnel か、研究室にある nginx を使います |

### 自動で行う (`scripts/setup_server.sh`)

下の手順 2〜7 (仮想環境、パッケージ、研究のリポジトリの確認、重みのコピー、環境ファイル、通しの点検、systemd の設定ファイル) は、1 つのコマンドで行えます。**GPU のある機械で**、手順 1 (取得) のあとに実行してください。

```bash
cd demo_site
scripts/setup_server.sh \
  --mask-root /path/to/AntiCloning_mask \
  --from-hf   元のホスト:/path/to/hf \
  --from-hub  元のホスト:/path/to/torch_hub \
  --smoke --start
```

まず、`--dry-run` を付けると、何を実行するかだけを表示します (何も書き込みません)。完了まで、新しい環境で、約 6 分 (ダウンロードの速さによる) です。

| 指定 | 内容 |
|---|---|
| `--mask-root` | 研究のリポジトリ (`anticloning_mask/` を含むフォルダ)。省略すると、いくつかの場所を探す |
| `--from-hf`、`--from-hub` | 重みのコピー元 (パス、または `ホスト:パス`)。すでに重みがそろっていれば、省略できる |
| `--weights-dir` | 重みの置き場 (既定: リポジトリの 1 つ上の `weights/`。中に `hf/` と `torch_hub/` を作る) |
| `--venv`、`--python` | 仮想環境の場所 (既定: リポジトリの `venv`)、それを作る Python (既定: `python3.10` など) |
| `--cuda` | PyTorch の CUDA の版 (`cu126`、`cu128`、`cu129`)。既定は、ドライバから自動で選ぶ |
| `--smoke`、`--start`、`--port` | 最後に、通しの点検を行う、サーバを起動する、ポート (既定 8000) |
| `--skip-install`、`--skip-weights`、`--force-env`、`--allow-no-gpu` | パッケージの導入を飛ばす、重みを飛ばす、違う内容の `env.local` を上書きする (元のファイルは `.bak-日時` に残す)、GPU がなくても続ける (試験用) |

- **事前の点検:** OS (Linux の x86_64)、Python (3.9〜3.12。動作確認は 3.10)、空き容量 (15 GiB 以上)、GPU とドライバ。**GPU がなければ、理由を示して、ここで止まります** (敵対的音声の生成は、GPU がないと動かないため)。
- **仮想環境:** `python3 -m venv` が使えない環境 (`python3-venv` が入っていない) では、`get-pip.py` で pip を入れる方法に、自動で切り替えます。
- **PyTorch の版:** ドライバが対応する CUDA の版 (`nvidia-smi`) から選びます。12.8 以上は PyPI の既定 (CUDA 12.8 版)、12.6〜12.7 は `cu126`、12.0〜12.5 は `cu126` (CUDA 12 の互換機能。未確認)、12.0 未満は止まります。Blackwell 世代の GPU には、12.8 以上のドライバが要ります。入れたあとに、PyTorch が GPU を認識するかも確かめます。
- **一時領域:** pip のダウンロードの一時領域は、`/tmp` ではなく、リポジトリの `server/work/tmp/` に置きます (`/tmp` が小さい機械で失敗するため)。経過は `server/work/setup-pip.log` に残ります。
- **やり直し:** 済んでいる工程は、飛ばします。失敗したら、原因を直して、同じコマンドを、もう一度実行してください。すでにある `server/env.local` は、内容が違っても、上書きしません。
- **行わないこと:** HTTPS での公開 (手順 8)、`sudo` が要る作業 (systemd への登録は、置くコマンドを表示するだけ)、研究のリポジトリや重みの、取得元の用意。
- **確認の範囲:** 開発環境で、まっさらなコピーから、最後まで通しで確かめました (NG 0、API の通し試験 38 項目を含む)。GPU なし、古いドライバ、CUDA 12.6 の場合は、`nvidia-smi` の模擬で、動作を確かめた段階です (実機では未確認)。

### 手順 (手作業で行う場合)

1. **取得**: `git clone` して、公開するブランチをチェックアウトします
2. **Python の環境**: `python3 -m venv venv && venv/bin/pip install -r requirements.txt`。GPU とドライバが CUDA 12.8 の PyTorch に合わないときは、`requirements.txt` の説明のとおり、先に、合う版を入れます
3. **研究のリポジトリ**: 研究室のサーバにすでにあれば、その場所を使います
4. **環境ファイル**: `cp server/env.example server/env.local` として、`ANTICLONING_MASK_ROOT`、`PYTHON` (手順 2 の `venv/bin/python`)、`HF_HOME`、`TORCH_HUB_DIR` を書きます。`LIBRITTS_PARQUET_DIR` と `PYARROW_PATH` は、候補話者を作り直すときだけ要ります。`env.local` は git に入れません 音声合成の設定 (任意): `EXTRA_PYTHONPATH` (venv に入れられないパッケージの場所。import の探索の末尾に足す)、`TTS_DEVICE` (`auto`・`cuda`・`cpu`。既定は `auto`)、`TTS_THREADS` (日本語の CPU の推論の本数。既定は 4)、`TTS_JA_DIR` (日本語モデルの場所)、`TTS_DISABLE=1` (音声合成を使わない)。
5. **重みを置く**:

   ```bash
   scripts/copy_weights.sh --from-hf 元のホスト:/path/to/hf --from-hub 元のホスト:/path/to/torch_hub
   ```

   コピー元は、動作確認済みの環境の `HF_HOME` と `TORCH_HUB_DIR` です (ローカルのパスでも、`ホスト:パス` でもよい)。コピー先は、手順 4 の値です (`--to-hf` と `--to-hub` で変えられます)。必要なのは `tar` と `sha256sum` (コピー元が別のマシンなら `ssh` も) だけで、`rsync` は要りません。コピーしたあと、`server/weights/*.sha256` で照合し、一致しなければ失敗で終わります。すでに正しいものがあれば、その組は飛ばします。`--dry-run` でコピーするものだけを確かめられ、`--verify-only` で照合だけできます。コピーするのは、次の 2 組だけです:
   - x-vector: `<HF_HOME>/hub/models--speechbrain--spkrec-xvect-voxceleb/`
   - UTMOS: `<TORCH_HUB_DIR>/tarepan_SpeechMOS_v1.2.0/` と `<TORCH_HUB_DIR>/checkpoints/utmos22_strong_step7459_v1.pt`
6. **点検**: `scripts/check_env.sh --smoke`。足りないものが一覧になります。NG が 0 になるまで直します (`--smoke` は、合成した音声で、生成を 1 回、実際に通します)
7. **常駐**: systemd が使える環境では、`deploy/voice-demo.service` の雛形の `User` とパスを書き換えて、`/etc/systemd/system/` に置き、`systemctl enable --now voice-demo` します。使えない環境 (コンテナなど) では、`scripts/start.sh 8000` を使います (再起動のたびに、起動し直す必要があります)。サーバは、既定で `127.0.0.1` だけで待ち受けます
8. **HTTPS で公開**: 次のどちらかを使います。
   - **Cloudflare Tunnel** (`cloudflared`): サーバから外向きに接続するだけで、ポートを開けずに、HTTPS で公開できます。
     - 試す (アカウント不要): `scripts/tunnel.sh quick 8000` で、一時的な URL (`https://....trycloudflare.com`) が出ます。URL を知っている人は、誰でも、この処理 (GPU) を使えます。確認が終わったら、Ctrl-C で閉じてください
     - 本番 (名前つき): Cloudflare に登録したドメインが要ります。`deploy/cloudflared-config.example.yml` の手順で、トンネルを作り、`scripts/tunnel.sh run <トンネル名>` で起動します
     - `cloudflared` は、環境変数 `CLOUDFLARED`、`server/work/bin/cloudflared`、`PATH` の順に探します (公式の配布: https://github.com/cloudflare/cloudflared/releases)
   - **nginx** (研究室に、すでに HTTPS の入口がある場合): `deploy/nginx.example.conf` を `/etc/nginx/conf.d/` に置いて、`server_name`、証明書、中継先のポートを書き換えます。**接続元の IP を上書きする `proxy_set_header` は、必須です。** サーバは、`127.0.0.1` から来た接続の `CF-Connecting-IP` (なければ `X-Forwarded-For` の先頭) を、利用者の接続元として信用します (同じ接続元からの同時の処理を 1 件に絞るため)。nginx から中継する接続も `127.0.0.1` から来るので、利用者が自分で付けたヘッダが通ると、接続元を偽って、この制限を避けられます。雛形は、この 2 つのヘッダを、`$remote_addr` で上書きしています。サーバを `127.0.0.1` 以外で待ち受けさせる場合も、必ず前段を通してください

### 公開の前に確認すること

- **プライバシーの文面**: 録音がサーバに届くので、研究室の規定 (同意の取り方、保存、ログ) に合わせて、フッターの「プライバシー」の文面を確認してください。サーバのログ (`server/work/serve.log`) には、ジョブの ID、反復回数、所要時間だけが残り、録音と接続元の IP アドレスは書きません。nginx を使う場合、nginx のログは、既定で、IP アドレスを記録します
- **アクセスの制限**: 認証はありません。公開 URL を知っている人は、誰でも、GPU を使えます。制限は、接続元ごとに同時 1 件と、順番待ち 5 件までだけです。必要なら、Cloudflare Access や nginx の認証で絞ってください
- **GPU の共有**: 他の研究のジョブと GPU を共有する場合、このサーバの 1 件 (約 16 秒) が、待たされることがあります
- **未確認のこと**: 研究室のサーバでの動作 (GPU の世代、ドライバ、ネットワーク)、複数人の同時利用、新しい環境への `pip install` の実行 (依存関係の解決だけを確かめた)、nginx の設定の構文 (確かめた環境に nginx がなかった)


## 素材について

`server/data/pool.npz` (ターゲット候補の x-vector) は、`scripts/build_pool.sh` で作り直せます。`server/data/tts.json` は、モデルの語彙と、日本語の記号表です (学習時のものと同じでなければならないので、変更しないでください)。

攻撃の設定 (学習率、損失の重みなど) は、`server/core.py` の `ATTACK` にあります。Beta の β だけは、Beta 探索の採用値である 1.0 に揃えています (研究の設定ファイルは 0.5)。

## 制限

- 英語のモデル (SpeechT5) は英語専用です。録音は他の言語でも構いません。読み上げる言語は、入力した文章から決めます (日本語の文字が 1 字でもあれば日本語)。日本語のモデルが無い環境では、日本語は入力できても、読み上げられません
- 音声合成もサーバで行うので、ブラウザは、モデルを取得せず、メモリも多く使いません。その代わり、混み合うと Step 3 も順番待ちになり、入力した文章がサーバへ届きます (プライバシーの文面に記載)
- 日本語のモデルは、JVS の声に近い x-vector で、短い文を、同じ文の繰り返しにしてしまうことがあります。終わりの検出で止めていますが、注意が崩れて末尾に届かない場合には効きません。長い文では、途中の語句の抜けや繰り返しが起きやすいため、ポーズの位置で区間に分けて合成しています (上の「日本語のクローン」)。ポーズのない長い数字の列 (16 桁など) は、分けられないので、読み誤ることがあります
- SpeechT5 は推論時も dropout が有効なので、同じ入力でも合成結果と類似度は毎回変わります
- 日本語の話者 (JVS) の x-vector で英語を読ませると、まれに (評価した 96 組のうち 2 件) 文を繰り返します。英語には、日本語のような、文の終わりの検出がありません
- 敵対的音声の生成と、クローンの合成は、それぞれ、サーバで 1 件ずつ処理するため、混み合うと順番待ちになります。サーバに GPU と、研究のリポジトリ、重みが要ります
- 効果はホワイトボックス条件のものです。他の音声合成モデルへの保護効果は示していません
- 日本語のクローンでの保護効果は、系統立てた評価をしていません (評価したのは、日本語モデルの合成品質です)
- 日本語の折り返しは、`word-break: auto-phrase` で、文節の途中で切らないようにしています。この指定は、Chrome と Edge の 119 以降で効きます。効かないブラウザ (Safari の安定版、Firefox など) では、`site/js/site/linebreak.js` が、BudouX で文を文節に分けて `<wbr>` を入れ、同じ見え方にします (Chrome では、何もしません)。Safari 実機での確認は、未実施です

## クレジット

- 話者エンコーダ: [SpeechBrain spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) (Apache-2.0)
- 音声合成: [Microsoft SpeechT5](https://huggingface.co/microsoft/speecht5_tts) / [HiFi-GAN](https://huggingface.co/microsoft/speecht5_hifigan) (MIT)。日本語は、SpeechT5 を追加学習したモデル
- ターゲット話者の埋め込みと参照音声: LibriTTS (CC BY 4.0)。参照音声は、候補 24 人の発話を 16 kHz モノラルに変換し、音量を調整したもの
- 日本語の音素化: [Open JTalk](https://open-jtalk.sourceforge.net/) と NAIST Japanese Dictionary (いずれも修正 BSD)。[pyopenjtalk-plus](https://pypi.org/project/pyopenjtalk-plus/) 経由で、サーバが使用
- 日本語モデルの学習データ: [JVS コーパス](https://arxiv.org/abs/1908.06248) と [JSUT コーパス](https://arxiv.org/abs/1711.00354) (東京大学)、LibriTTS の一部 (CC BY 4.0)。**JVS と JSUT の音声の利用は、学術機関の研究、非営利の研究 (企業内を含む)、個人利用に限られ、再配布は認められていません。** このデモの日本語モデルは、その条件に従い、非営利の研究・個人利用に限って使ってください
- 音質の評価: [UTMOS](https://github.com/tarepan/SpeechMOS) (SpeechMOS、MIT)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) (MIT)、[Inter](https://github.com/rsms/inter) (OFL-1.1)、[Lucide](https://lucide.dev/) (ISC)、[BudouX](https://github.com/google/budoux) 0.10.1 (Apache-2.0。Safari など、`word-break: auto-phrase` が使えないブラウザで、日本語を文節の途中で折り返さないために使う。`site/vendor/budoux/` に、ライセンスとともに同梱)
