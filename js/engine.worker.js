// 重い処理をすべて引き受けるワーカー。音声はこのワーカーとメインスレッドの間でしか受け渡さない。
import * as ort from "../vendor/ort/ort.wasm.bundle.min.mjs";
import { SignalChain } from "./dsp.js";
import { MaskAttack, cosine } from "./attack.js";
import { Cancelled, SpeechT5, encodeText, normalizeText } from "./tts.js";

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

let constants, chain, xvector, tts;
let cancelled = false;
const session = {};

const isCancelled = () => cancelled;
const nextTick = () => new Promise((resolve) => setTimeout(resolve));
const tensor = (data, dims) => new ort.Tensor("float32", data, dims);

async function init() {
  if (xvector) return { threads: ort.env.wasm.numThreads };
  constants = await (await fetch(new URL("assets/constants.json", base))).json();
  constants.targets.forEach((t) => (t.emb = Float32Array.from(t.emb)));
  chain = new SignalChain(constants);
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

function unit(e) {
  let n = 0;
  for (let i = 0; i < e.length; i++) n += e[i] * e[i];
  n = Math.sqrt(n);
  return e.map((v) => v / n);
}

async function clone({ text }, progress) {
  await init();
  if (!session.eProtected) throw new Error("先に敵対的音声を生成してください。");
  const norm = normalizeText(text, constants.tts.vocab);
  if (norm.error) throw new Error(norm.error);
  await loadTts((p) => progress({ kind: "load", ...p }));
  progress({ kind: "encode", text: norm.text });
  await nextTick();
  const encoded = await tts.encode(encodeText(norm.text, constants.tts.vocab));
  const results = {};
  const variants = [
    { key: "clean", speaker: session.eSrc },
    { key: "protected", speaker: session.eProtected },
  ];
  for (const v of variants) {
    const wave = await tts.decode(encoded, unit(v.speaker), {
      isCancelled,
      onStep: (step, maxSteps) => progress({ kind: "decode", variant: v.key, step, maxSteps, chars: norm.text.length }),
    });
    const e = await embed(wave);
    const result = { wave, cosSrc: cosine(e, session.eSrc), cosTgt: cosine(e, session.target.emb) };
    results[v.key] = result;
    progress({ kind: "variant", variant: v.key, cosSrc: result.cosSrc, cosTgt: result.cosTgt, wave: wave.slice() });
    await nextTick();
  }
  return { data: { text: norm.text, clean: strip(results.clean), protected: strip(results.protected) } };
}

const strip = ({ cosSrc, cosTgt }) => ({ cosSrc, cosTgt });

const jobs = {
  init: async () => ({ data: await init() }),
  attack,
  clone,
  normalize: async ({ text }) => {
    await init();
    return { data: normalizeText(text, constants.tts.vocab) };
  },
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
