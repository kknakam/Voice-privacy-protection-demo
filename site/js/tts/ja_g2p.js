// OpenJTalk のフルコンテキストラベルから、日本語モデルの入力 (音素 + 韻律記号) の ID 列を作る。
// 記号の付け方は学習時の ja_frontend.py (栗原らの方式) と同じ。
//   ^ 文頭   $ 平叙文の文末   ? 疑問文の文末   _ ポーズ
//   # アクセント句の境界   [ ピッチの上昇   ] ピッチの下降 (アクセント核の直後)

const VOWELS = new Set(["a", "i", "u", "e", "o", "N", "cl"]);

function num(re, s) {
  const m = re.exec(s);
  return m === null ? -50 : parseInt(m[1], 10); // 値が "xx" の場合は -50
}

// ラベルの配列 -> 記号の配列
export function labelsToSymbols(labels) {
  const out = [];
  const n = labels.length;
  for (let i = 0; i < n; i++) {
    const lab = labels[i];
    let p3 = /-(.*?)\+/.exec(lab)[1];
    if (["A", "I", "U", "E", "O"].includes(p3)) p3 = p3.toLowerCase(); // 無声化母音は通常の母音として扱う
    if (p3 === "sil") {
      if (i === 0) out.push("^");
      else out.push(num(/!(\d+)_/, lab) === 1 ? "?" : "$"); // 末尾: 疑問文かどうか
      continue;
    }
    if (p3 === "pau") {
      out.push("_");
      continue;
    }
    out.push(p3);
    if (i + 1 >= n) continue;
    const a1 = num(/\/A:([0-9-]+)\+/, lab); // アクセント核からの相対位置
    const a2 = num(/\+(\d+)\+/, lab); // アクセント句内の前からのモーラ位置
    const a3 = num(/\+(\d+)\//, lab); // アクセント句内の後ろからのモーラ位置
    const f1 = num(/\/F:(\d+)_/, lab); // アクセント句のモーラ数
    const a2Next = num(/\+(\d+)\+/, labels[i + 1]);
    if (a3 === 1 && a2Next === 1 && VOWELS.has(p3)) out.push("#");
    else if (a1 === 0 && a2Next === a2 + 1 && a2 !== f1) out.push("]");
    else if (a2 === 1 && a2Next === 2) out.push("[");
  }
  return out;
}

// 記号の配列 -> モデルの入力 ID (末尾に </s>)。表にない記号があれば null
export function symbolsToIds(symbols, ja) {
  const ids = [];
  for (const s of symbols) {
    const k = ja.symbols.indexOf(s);
    if (k < 0) return null;
    ids.push(ja.baseVocab + k);
  }
  ids.push(ja.eos);
  return ids;
}
