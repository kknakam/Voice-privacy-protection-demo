// 「研究のシナリオ」の図 3: マスクの掛け算の模式図。実際の音声ではなく、見せるための決まった模様を描く。
// 装飾なので、読み上げの対象外 (aria-hidden)。意味は、図の説明文 (aria-label と figcaption) が伝える。

const COLS = 12;
const ROWS = 8;

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// 低い周波数ほど強く、時間とともに強弱が変わる (声のスペクトログラムのつもり)。r = 0 が最も低い周波数
function spectrogramLevel(r, c) {
  const decay = 1 - r / (ROWS + 2);
  const wave = 0.55 + 0.45 * Math.sin(c * 0.85 + r * 0.6) * Math.cos(r * 1.3 - c * 0.2);
  return clamp(Math.round(decay * wave * 4.4), 0, 4);
}

function build() {
  const host = document.getElementById("mask-demo");
  if (!host) return;

  let seed = 20261118;
  const rnd = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };

  const source = [];
  const keep = [];
  const result = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const level = spectrogramLevel(r, c);
      // いちばん強い成分 (level 4) は削らない。中間の周波数の、強さが中くらいのマス目を、ところどころ削る
      const cut = level >= 1 && level <= 3 && r >= 1 && r <= 5 && rnd() < 0.5;
      source.push(level);
      keep.push(!cut);
      result.push(cut ? 0 : level);
    }
  }

  const panel = (caption, classOf) => {
    const wrap = document.createElement("div");
    wrap.className = "cells-panel";
    const grid = document.createElement("div");
    grid.className = "cells";
    grid.setAttribute("aria-hidden", "true");
    for (let r = ROWS - 1; r >= 0; r--) {   // 上が高い周波数
      for (let c = 0; c < COLS; c++) {
        const cell = document.createElement("span");
        cell.className = `cell ${classOf(r * COLS + c)}`;
        grid.appendChild(cell);
      }
    }
    const label = document.createElement("span");
    label.className = "cells-caption";
    label.setAttribute("aria-hidden", "true");
    label.textContent = caption;
    wrap.append(grid, label);
    return wrap;
  };

  const op = (text) => {
    const span = document.createElement("span");
    span.className = "op";
    span.setAttribute("aria-hidden", "true");
    span.textContent = text;
    return span;
  };

  host.replaceChildren(
    panel("元の音声", (i) => `cell-${source[i]}`),
    op("×"),
    panel("マスク", (i) => (keep[i] ? "mask-keep" : "mask-cut")),
    op("="),
    panel("結果", (i) => `cell-${result[i]}`),
  );
}

build();
