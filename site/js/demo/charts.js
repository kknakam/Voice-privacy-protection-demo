// 依存なしの小さな描画: ヒートマップ (スペクトログラム、マスク)、波形、ツールチップ。
// ラベルは textContent で入れ、色は系列のマークだけに使う。

class Tooltip {
  constructor(host) {
    this.host = host;
    this.node = document.createElement("div");
    this.node.className = "chart-tooltip";
    this.node.hidden = true;
    host.appendChild(this.node);
  }

  // rows: [{ value, label, color? }]
  show(x, y, title, rows) {
    this.node.replaceChildren();
    if (title) {
      const head = document.createElement("div");
      head.className = "chart-tooltip-title";
      head.textContent = title;
      this.node.appendChild(head);
    }
    for (const row of rows) {
      const line = document.createElement("div");
      line.className = "chart-tooltip-row";
      if (row.color) {
        const key = document.createElement("span");
        key.className = "line-key";
        key.style.background = row.color;
        line.appendChild(key);
      }
      const value = document.createElement("strong");
      value.textContent = row.value;
      const label = document.createElement("span");
      label.textContent = row.label;
      line.append(value, label);
      this.node.appendChild(line);
    }
    this.node.hidden = false;
    const w = this.node.offsetWidth, h = this.node.offsetHeight;
    const W = this.host.clientWidth;
    const left = x + 12 + w > W ? Math.max(0, x - w - 12) : x + 12;
    this.node.style.transform = `translate(${left}px, ${Math.max(0, y - h - 8)}px)`;
  }

  hide() {
    this.node.hidden = true;
  }
}

// 単一色相の順序スケール (紙の色 -> 濃い藍)
const STOPS = [[246, 245, 244], [139, 150, 214], [33, 49, 131]];
export const HEAT_LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (STOPS.length - 1);
    const a = Math.min(Math.floor(t), STOPS.length - 2);
    const f = t - a;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = STOPS[a][c] + (STOPS[a + 1][c] - STOPS[a][c]) * f;
  }
  return lut;
})();

// values は [T * F]。低い周波数が下になるように描く。
export function drawHeatmap(canvas, { values, F, T }, { lo, hi, invert = false }) {
  canvas.width = T;
  canvas.height = F;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(T, F);
  const d = image.data;
  const scale = 255 / (hi - lo);
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < F; k++) {
      let u = (values[t * F + k] - lo) * scale;
      if (invert) u = 255 - u;
      const i = Math.max(0, Math.min(255, Math.round(u))) * 3;
      const o = ((F - 1 - k) * T + t) * 4;
      d[o] = HEAT_LUT[i];
      d[o + 1] = HEAT_LUT[i + 1];
      d[o + 2] = HEAT_LUT[i + 2];
      d[o + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
}

// ヒートマップのセル読み取り用ツールチップ
export function attachHeatTooltip(figure, canvas, { values, F, T }, { sampleRate, hop, format, label }) {
  // 再生成のたびに呼ばれるので、前回のツールチップとリスナーを外す
  figure.heatTip?.abort.abort();
  figure.heatTip?.tip.node.remove();
  const tip = new Tooltip(figure);
  const abort = new AbortController();
  figure.heatTip = { tip, abort };
  const { signal } = abort;
  canvas.addEventListener("pointermove", (e) => {
    const r = canvas.getBoundingClientRect();
    const fr = figure.getBoundingClientRect();
    const t = Math.min(T - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * T)));
    const k = Math.min(F - 1, Math.max(0, F - 1 - Math.floor(((e.clientY - r.top) / r.height) * F)));
    const sec = (t * hop) / sampleRate;
    const hz = (k * sampleRate) / (2 * (F - 1));
    tip.show(e.clientX - fr.left, e.clientY - fr.top, `${sec.toFixed(2)} 秒 / ${Math.round(hz).toLocaleString()} Hz`, [
      { value: format(values[t * F + k]), label },
    ]);
  }, { signal });
  canvas.addEventListener("pointerleave", () => tip.hide(), { signal });
}

export function drawWaveform(canvas, samples, color) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, h = canvas.clientHeight || 72;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = color;
  const bar = 2, gap = 2;
  const n = Math.floor(w / (bar + gap));
  const per = samples.length / n;
  let top = 1e-6;
  const peaks = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let p = 0;
    const end = Math.min(samples.length, Math.floor((i + 1) * per));
    for (let j = Math.floor(i * per); j < end; j++) p = Math.max(p, Math.abs(samples[j]));
    peaks[i] = p;
    top = Math.max(top, p);
  }
  for (let i = 0; i < n; i++) {
    const bh = Math.max(2, (peaks[i] / top) * (h - 4));
    ctx.fillRect(i * (bar + gap), (h - bh) / 2, bar, bh);
  }
}
