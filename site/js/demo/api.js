// サーバ (server/app.py) との受け渡し。敵対的音声の生成 (Step 2) と、クローンの合成 (Step 3) は、どちらも、サーバで行う。
// 録音と、入力した文章は、同じ出どころの /api/ へ送る
const base = "api/";

async function json(res) {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `サーバからの応答が正しくありません (HTTP ${res.status})`);
  return body;
}

// サーバの状態。{ ok, queued, tts: { state: "loading" | "ready", en, ja } }。静的な配信だけのサーバ (API がない) では null
export async function health() {
  try {
    const res = await fetch(`${base}health`, { cache: "no-store" });
    const body = res.ok ? await res.json() : null;
    return body?.ok === true ? body : null;
  } catch {
    return null;
  }
}

async function postJson(path, body) {
  return json(await fetch(`${base}${path}`, { method: "POST", body: JSON.stringify(body), headers: { "Content-Type": "application/json" } }));
}

// 入力中の案内 (モデルは使わない)。{ lang, text } または { lang, error }
export const normalize = (text) => postJson("normalize", { text });

// クローンの合成を頼む。emb は、敵対的音声を作ったときにサーバが返した x-vector { src, adv, tgt }
export async function submitClone(text, emb) {
  return (await postJson("clones", { text, emb: { src: emb.src, adv: emb.adv, tgt: emb.tgt } })).id;
}

export async function submit(blob, steps) {
  const res = await fetch(`${base}jobs?steps=${steps}`, { method: "POST", body: blob, headers: { "Content-Type": "audio/wav" } });
  return (await json(res)).id;
}

export const status = async (id) => json(await fetch(`${base}jobs/${id}`, { cache: "no-store" }));
export const result = async (id) => json(await fetch(`${base}jobs/${id}/result`, { cache: "no-store" }));
export const cancel = (id) => fetch(`${base}jobs/${id}`, { method: "DELETE" }).catch(() => {});

const bytes = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
const pcmToFloat = (b64) => Float32Array.from(new Int16Array(bytes(b64).buffer), (v) => v / 32768);

// クローンの結果を、画面が使う形にする。{ text, lang, segments, clean: { cosSrc, cosTgt, wave }, protected: { ... } }
export function decodeClone(raw) {
  const variant = ({ cosSrc, cosTgt, pcm16 }) => ({ cosSrc, cosTgt, wave: pcmToFloat(pcm16) });
  return { text: raw.text, lang: raw.lang, segments: raw.segments, seconds: raw.seconds, clean: variant(raw.clean), protected: variant(raw.protected) };
}

// 量子化 (0〜255) して送られたスペクトログラムとマスクを、値に戻す。並びは [T * F] (時間が外側)
function dequantize({ lo, hi, q }, F, T) {
  const raw = bytes(q);
  const values = new Float32Array(raw.length);
  const scale = (hi - lo) / 255;
  for (let i = 0; i < raw.length; i++) values[i] = lo + raw[i] * scale;
  return { values, F, T };
}

// 敵対的音声の結果を、画面が使う形にする
export function decode(raw) {
  const pcm = new Int16Array(bytes(raw.adv.pcm16).buffer);
  const protectedWave = Float32Array.from(pcm, (v) => v / 32768);
  const { F, T } = raw.heat;
  return {
    protected: protectedWave,
    target: raw.target,
    stats: raw.stats,
    seconds: raw.seconds,
    emb: raw.emb,
    clean: dequantize(raw.heat.clean, F, T),
    adv: dequantize(raw.heat.adv, F, T),
    mask: dequantize(raw.heat.mask, F, T),
  };
}
