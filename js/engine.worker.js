// 重い処理をすべて引き受けるワーカー。音声はこのワーカーとメインスレッドの間でしか受け渡さない。
import * as ort from "../vendor/ort/ort.wasm.bundle.min.mjs";
import { SignalChain } from "./dsp.js";
import { MaskAttack, cosine } from "./attack.js";
import { Cancelled, SpeechT5, encodeText, normalizeText } from "./tts.js";
import { SpeechT5Ja, normalizeJa } from "./tts_ja.js";
import { JaFrontend } from "./ja_frontend.js";

const base = new URL("../", import.meta.url);
// バンドル版は JS のグルーを内蔵しているので、wasm の場所だけを指定する
ort.env.wasm.wasmPaths = { wasm: new URL("vendor/ort/ort-wasm-simd-threaded.wasm", base).href };
ort.env.wasm.numThreads = self.crossOriginIsolated
  ? Math.min(4, Math.max(1, Math.ceil((navigator.hardwareConcurrency || 2) / 2)))
  : 1;

// デコーダは量子化すると声の再現性が大きく落ちるため fp16 を使う。
// エンコーダは bnb4 でも再現性と発音が変わらないので小さい版を使う (README 参照)
const TTS_FILES = [
  { key: "encoder", file: "encoder_model_bnb4.onnx", mib: 48.4, level: "all" },
  { key: "decoder", file: "decoder_model_merged_fp16.onnx", mib: 114.0, level: "all" },
  { key: "vocoder", file: "decoder_postnet_and_vocoder_fp16.onnx", mib: 26.5, level: "all" },
];
const MODEL_CACHE = "live-demo-models-v1";

let constants, chain, xvector, tts, ttsJa, jaManifest, jaFront;
let cancelled = false;
const session = {};

const isCancelled = () => cancelled;
const nextTick = () => new Promise((resolve) => setTimeout(resolve));
const tensor = (data, dims) => new ort.Tensor("float32", data, dims);

async function loadConstants() {
  if (constants) return;
  constants = await (await fetch(new URL("assets/constants.json", base))).json();
  constants.targets.forEach((t) => (t.emb = Float32Array.from(t.emb)));
  chain = new SignalChain(constants);
}

async function init() {
  if (xvector) return { threads: ort.env.wasm.numThreads };
  await loadConstants();
  xvector = await ort.InferenceSession.create(new URL("models/xvector_grad.onnx", base).href, {
    executionProviders: ["wasm"],
  });
  return { threads: ort.env.wasm.numThreads };
}

async function runXvector(feats, Tf, eSrc, eTgt, wAway, wToward) {
  const r = await xvector.run({
    feats: tensor(feats, [1, Tf, chain.nMels]),
    e_src: tensor(eSrc, [1, 512]),
    e_tgt: tensor(eTgt, [1, 512]),
    w_away: tensor(new Float32Array([wAway]), [1, 1]),
    w_toward: tensor(new Float32Array([wToward]), [1, 1]),
    tau: tensor(new Float32Array([constants.attack.tau]), [1, 1]),
  });
  return { loss: r.loss.data[0], emb: r.emb.data, grad: r.grad.data };
}

async function embed(wave) {
  const fb = chain.fbank(wave);
  const zero = constants.targets[0].emb;
  return Float32Array.from((await runXvector(fb.feats, fb.Tf, zero, zero, 0, 0)).emb);
}

async function attack({ wave, steps, target: targetId }, progress) {
  await init();
  const target = constants.targets.find((t) => t.id === targetId);
  if (!target) throw new Error("ターゲット話者を選んでください。");
  const eSrc = await embed(wave);
  const cfg = constants.attack;
  const job = new MaskAttack({
    wave, chain, cfg,
    gradFn: (feats, Tf) => runXvector(feats, Tf, eSrc, target.emb, cfg.wAway, cfg.wToward),
  });
  const started = performance.now();
  for (let step = 1; step <= steps; step++) {
    if (cancelled) throw new Cancelled();
    await job.step();
    progress({ step, steps, msPerStep: (performance.now() - started) / step });
    await nextTick();
  }
  const protectedWave = Float32Array.from(job.xAdv);
  session.eSrc = eSrc;
  session.eProtected = await embed(protectedWave);
  session.target = target;
  const clean = chain.logMagnitude(wave);
  const adv = chain.logMagnitude(protectedWave);
  const mask = Float32Array.from(job.mask);
  return {
    data: {
      protected: protectedWave,
      target: { id: target.id, speaker: target.speaker },
      snr: job.snrDb(),
      stats: job.maskStats(),
      msPerStep: (performance.now() - started) / steps,
      mask: { values: mask, F: job.spec.F, T: job.spec.T },
      clean,
      adv,
    },
    transfer: [protectedWave.buffer, mask.buffer, clean.values.buffer, adv.values.buffer],
  };
}

async function fetchCached(url, onBytes) {
  const cache = await caches.open(MODEL_CACHE).catch(() => null);
  const hit = cache && (await cache.match(url));
  if (hit) {
    const bytes = new Uint8Array(await hit.arrayBuffer());
    onBytes(bytes.length, true);
    return bytes;
  }
  const res = await fetch(url, { mode: "cors" });
  if (!res.ok) throw new Error(`モデルの取得に失敗しました (HTTP ${res.status})`);
  const chunks = [];
  let received = 0;
  const reader = res.body.getReader();
  for (;;) {
    if (cancelled) {
      await reader.cancel();
      throw new Cancelled();
    }
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onBytes(value.length, false);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) {
    bytes.set(c, offset);
    offset += c.length;
  }
  // 容量不足でキャッシュできなくても処理は続ける
  if (cache) await cache.put(url, new Response(bytes)).catch(() => {});
  return bytes;
}

// 版番号が違う models_ja/ の保存済みデータ (差し替え前の古いモデル) を消す。容量を使い続けないようにするため
async function purgeOldModels(version) {
  const cache = await caches.open(MODEL_CACHE).catch(() => null);
  if (!cache) return;
  const keep = `?v=${version}`;
  for (const request of await cache.keys().catch(() => [])) {
    if (request.url.includes("/models_ja/") && !request.url.endsWith(keep)) await cache.delete(request).catch(() => {});
  }
}

// models_ja/ のファイルを取得する。Cloudflare Pages の 1 ファイル 25 MiB の上限に合わせ、大きなファイルは分割して置いてある
// (manifest の parts)。順に取得してつなぎ、大きさと sha256 を確かめる。
// URL に版番号 (?v=) を付けるので、モデルを差し替えても、ブラウザに保存済みの古いデータは使われない
async function fetchAssembled(dir, entry, version, onBytes) {
  const urlOf = (name) => new URL(`models_ja/${dir}${name}?v=${version}`, base).href;
  const urls = (entry.parts ?? [entry.file]).map(urlOf);
  let bytes;
  if (!entry.parts) {
    bytes = await fetchCached(urls[0], onBytes);
  } else {
    bytes = new Uint8Array(entry.bytes);
    let offset = 0;
    for (const url of urls) {
      const part = await fetchCached(url, onBytes);
      if (offset + part.length > bytes.length) throw new Error(`${entry.file}: 取得したデータが大きすぎます。`);
      bytes.set(part, offset);
      offset += part.length;
    }
    if (offset !== bytes.length) bytes = bytes.subarray(0, offset); // 足りない場合は、大きさの確認で失敗させる
  }
  let problem = "";
  if (entry.bytes !== undefined && bytes.length !== entry.bytes) problem = "大きさが合いません";
  else if (entry.sha256 && globalThis.crypto?.subtle) {
    const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
    if (Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("") !== entry.sha256) problem = "内容が合いません";
  }
  if (problem) {
    // 壊れたデータを保存したままにしないよう、消してから失敗させる
    const cache = await caches.open(MODEL_CACHE).catch(() => null);
    if (cache) await Promise.all(urls.map((u) => cache.delete(u).catch(() => {})));
    throw new Error(`${entry.file} の取得に失敗しました (${problem})。ページを再読み込みして、もう一度お試しください。`);
  }
  return bytes;
}

async function loadTts(progress) {
  if (tts) return;
  const { repo, revision } = constants.tts;
  const total = TTS_FILES.reduce((n, f) => n + f.mib, 0);
  let loaded = 0;
  const sessions = {};
  for (const f of TTS_FILES) {
    const url = `https://huggingface.co/${repo}/resolve/${revision}/onnx/${f.file}`;
    const scale = 2 ** 20;
    const bytes = await fetchCached(url, (n, fromCache) => {
      loaded += n / scale;
      progress({ phase: "download", loadedMiB: loaded, totalMiB: total, fromCache });
    });
    progress({ phase: "prepare", name: f.key });
    await nextTick();
    sessions[f.key] = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: f.level,
    });
  }
  tts = new SpeechT5(ort, sessions);
}

// 日本語モデルは models_ja/ に置いたときだけ使える (重みは公開リポジトリに含めない)。manifest.json があるかで判定する
async function jaStatus() {
  await loadConstants();
  if (jaManifest === undefined) {
    jaManifest = null;
    try {
      const res = await fetch(new URL("models_ja/manifest.json", base));
      if (res.ok) {
        const m = await res.json();
        if (Array.isArray(m.files) && constants.ttsJa) jaManifest = m;
      }
    } catch {
      // 置かれていない場合は日本語を出さない
    }
  }
  return {
    data: {
      available: Boolean(jaManifest),
      freeInput: Boolean(jaManifest?.frontend),
      presets: jaManifest ? constants.ttsJa.presets.map((p) => p.text) : [],
    },
  };
}

// 自由入力のための音素化 (OpenJTalk の WASM と辞書)。初めて定型文以外が入力されたときに読み込む
async function loadFrontend(progress) {
  if (jaFront) return;
  const fe = jaManifest?.frontend;
  if (!fe) throw new Error("日本語の音素化の辞書が見つかりません。");
  const total = fe.files.reduce((n, f) => n + f.mib, 0);
  let loaded = 0;
  const createOpenJTalk = (await import("../vendor/openjtalk/jt.mjs")).default;
  // OpenJTalk が変換できない入力で出す警告 (標準エラー出力) は、画面のエラーとして扱うのでコンソールには出さない
  const module = await createOpenJTalk({ print: () => {}, printErr: () => {} });
  module.FS.mkdir("/dic");
  for (const f of fe.files) {
    const bytes = await fetchAssembled(fe.dir, f, jaManifest.version ?? 0, (n, fromCache) => {
      loaded += n / 2 ** 20;
      progress({ phase: "download", loadedMiB: loaded, totalMiB: total, fromCache, what: "日本語の辞書" });
    });
    module.FS.writeFile(`/dic/${f.file}`, bytes);
  }
  progress({ phase: "prepare", name: "frontend" });
  await nextTick();
  const front = new JaFrontend(module);
  if (!front.ready) throw new Error("日本語の辞書を読み込めませんでした。");
  jaFront = front;
}

// 日本語の文章 -> { text, ids }。定型文は計算済みの ID、それ以外は OpenJTalk (WASM) で作る
async function japaneseIds(text, progress) {
  const norm = normalizeJa(text, constants.ttsJa, Boolean(jaManifest?.frontend));
  if (norm.error) throw new Error(norm.error);
  if (norm.ids) return norm;
  await loadFrontend((p) => progress({ kind: "load", ...p }));
  const r = jaFront.encode(norm.text, constants.ttsJa);
  if (r.error) throw new Error(r.error);
  return { text: norm.text, ids: r.ids };
}

async function loadTtsJa(progress) {
  if (ttsJa) return;
  if (!jaManifest) throw new Error("日本語の音声合成モデルが見つかりません。");
  await purgeOldModels(jaManifest.version ?? 0);
  const total = jaManifest.files.reduce((n, f) => n + f.mib, 0);
  let loaded = 0;
  const sessions = {};
  for (const f of jaManifest.files) {
    const bytes = await fetchAssembled("", f, jaManifest.version ?? 0, (n, fromCache) => {
      loaded += n / 2 ** 20;
      progress({ phase: "download", loadedMiB: loaded, totalMiB: total, fromCache });
    });
    progress({ phase: "prepare", name: f.key });
    await nextTick();
    sessions[f.key] = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  }
  ttsJa = new SpeechT5Ja(ort, sessions);
}

function unit(e) {
  let n = 0;
  for (let i = 0; i < e.length; i++) n += e[i] * e[i];
  n = Math.sqrt(n);
  return e.map((v) => v / n);
}

async function clone({ text, lang }, progress) {
  await init();
  if (!session.eProtected) throw new Error("先に敵対的音声を生成してください。");
  const ja = lang === "ja";
  let shown, chars, expected, model, encoded;
  if (ja) {
    await jaStatus();
    const norm = await japaneseIds(text, progress);
    await loadTtsJa((p) => progress({ kind: "load", ...p }));
    progress({ kind: "encode", text: norm.text });
    await nextTick();
    model = ttsJa;
    encoded = await ttsJa.encode(norm.ids);
    shown = norm.text;
    chars = norm.text.length;
    expected = norm.ids.length * 2; // 実測で 1 トークンあたり約 2 ステップ
  } else {
    const norm = normalizeText(text, constants.tts.vocab);
    if (norm.error) throw new Error(norm.error);
    await loadTts((p) => progress({ kind: "load", ...p }));
    progress({ kind: "encode", text: norm.text });
    await nextTick();
    model = tts;
    encoded = await tts.encode(encodeText(norm.text, constants.tts.vocab));
    shown = norm.text;
    chars = norm.text.length;
  }
  const results = {};
  const variants = [
    { key: "clean", speaker: session.eSrc },
    { key: "protected", speaker: session.eProtected },
  ];
  for (const v of variants) {
    const wave = await model.decode(encoded, unit(v.speaker), {
      isCancelled,
      onStep: (step, maxSteps) => progress({ kind: "decode", variant: v.key, step, maxSteps, chars, expected }),
    });
    const e = await embed(wave);
    const result = { wave, cosSrc: cosine(e, session.eSrc), cosTgt: cosine(e, session.target.emb) };
    results[v.key] = result;
    progress({ kind: "variant", variant: v.key, cosSrc: result.cosSrc, cosTgt: result.cosTgt, wave: wave.slice() });
    await nextTick();
  }
  return { data: { text: shown, clean: strip(results.clean), protected: strip(results.protected) } };
}

const strip = ({ cosSrc, cosTgt }) => ({ cosSrc, cosTgt });

const jobs = {
  init: async () => ({ data: await init() }),
  attack,
  clone,
  normalize: async ({ text, lang }) => {
    await init();
    if (lang === "ja") {
      await jaStatus();
      const r = normalizeJa(text, constants.ttsJa, Boolean(jaManifest?.frontend));
      return { data: r.error ? { error: r.error } : { text: r.text } };
    }
    return { data: normalizeText(text, constants.tts.vocab) };
  },
  jaStatus,
};

let queue = Promise.resolve();

self.onmessage = (e) => {
  const { id, type, payload } = e.data;
  if (type === "cancel") {
    cancelled = true;
    return;
  }
  const progress = (data) => self.postMessage({ id, kind: "progress", data });
  queue = queue.then(async () => {
    cancelled = false;
    try {
      const { data, transfer = [] } = await jobs[type](payload, progress);
      self.postMessage({ id, kind: "result", data }, transfer);
    } catch (err) {
      if (err instanceof Cancelled) self.postMessage({ id, kind: "cancelled" });
      else self.postMessage({ id, kind: "error", message: err?.message || String(err) });
    }
  });
};
