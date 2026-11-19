// 敵対的音声を作るサーバ (server/app.py) との受け渡し。録音は、同じ出どころの /api/ へ送る
const base = "api/";

async function json(res) {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `サーバからの応答が正しくありません (HTTP ${res.status})`);
  return body;
}

// サーバに、この処理があるか (静的な配信だけのサーバでは、ない)
export async function available() {
  try {
    const res = await fetch(`${base}health`, { cache: "no-store" });
    return res.ok && (await res.json()).ok === true;
  } catch {
    return false;
  }
}

export async function submit(blob, steps) {
  const res = await fetch(`${base}jobs?steps=${steps}`, { method: "POST", body: blob, headers: { "Content-Type": "audio/wav" } });
  return (await json(res)).id;
}

export const status = async (id) => json(await fetch(`${base}jobs/${id}`, { cache: "no-store" }));
export const result = async (id) => json(await fetch(`${base}jobs/${id}/result`, { cache: "no-store" }));
export const cancel = (id) => fetch(`${base}jobs/${id}`, { method: "DELETE" }).catch(() => {});

const bytes = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

// 量子化 (0〜255) して送られたスペクトログラムとマスクを、値に戻す。並びは [T * F] (時間が外側)
function dequantize({ lo, hi, q }, F, T) {
  const raw = bytes(q);
  const values = new Float32Array(raw.length);
  const scale = (hi - lo) / 255;
  for (let i = 0; i < raw.length; i++) values[i] = lo + raw[i] * scale;
  return { values, F, T };
}

// サーバの結果を、画面とワーカーが使う形にする
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
