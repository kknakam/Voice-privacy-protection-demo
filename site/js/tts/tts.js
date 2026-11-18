// SpeechT5 (ONNX) による音声合成と、入力文の正規化。
// 語彙は英語の文字と一部の記号だけで数字を含まないため、数字は英単語へ展開する。

export const MAX_INPUT_CHARS = 200;

const ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
  "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
const SCALES = ["", "thousand", "million", "billion", "trillion"];
const ORDINAL_FIX = { one: "first", two: "second", three: "third", five: "fifth", eight: "eighth", nine: "ninth", twelve: "twelfth" };

function belowThousand(n) {
  const words = [];
  if (n >= 100) {
    words.push(ONES[Math.floor(n / 100)], "hundred");
    n %= 100;
  }
  if (n >= 20) {
    const t = TENS[Math.floor(n / 10)];
    words.push(n % 10 ? `${t}-${ONES[n % 10]}` : t);
  } else if (n > 0) {
    words.push(ONES[n]);
  }
  return words.join(" ");
}

export function integerToWords(digits) {
  const clean = digits.replace(/^0+(?=\d)/, "");
  if (clean === "0") return "zero";
  if (clean.length > 15) return [...clean].map((d) => ONES[+d]).join(" ");
  const groups = [];
  for (let end = clean.length; end > 0; end -= 3) groups.push(+clean.slice(Math.max(0, end - 3), end));
  const words = [];
  for (let i = groups.length - 1; i >= 0; i--) {
    if (groups[i]) words.push(belowThousand(groups[i]) + (SCALES[i] ? ` ${SCALES[i]}` : ""));
  }
  return words.join(" ");
}

function numberToWords(token) {
  const [whole, frac] = token.replace(/,/g, "").split(".");
  const head = integerToWords(whole || "0");
  return frac ? `${head} point ${[...frac].map((d) => ONES[+d]).join(" ")}` : head;
}

function ordinalWords(digits) {
  const words = integerToWords(digits);
  const m = words.match(/^(.*?)([a-z]+)$/);
  const [, prefix, last] = m;
  if (ORDINAL_FIX[last]) return prefix + ORDINAL_FIX[last];
  return prefix + (last.endsWith("y") ? `${last.slice(0, -1)}ieth` : `${last}th`);
}

// 入力文を語彙に含まれる文字だけの文へ変換する。{ text } または { error } を返す
export function normalizeText(raw, vocab) {
  let s = raw.normalize("NFKC").trim();
  if (!s) return { error: "文章を入力してください。" };
  if (raw.length > MAX_INPUT_CHARS) return { error: `${MAX_INPUT_CHARS} 文字以内で入力してください。` };
  const foreign = s.match(/\p{L}/gu)?.find((c) => !/\p{Script=Latin}/u.test(c));
  if (foreign) return { error: `SpeechT5 は英語専用のため、英語で入力してください（使用できない文字: ${foreign}）。` };

  s = s.replace(/[‘’ʼ]/g, "'").replace(/[“”]/g, '"').replace(/…/g, "...")
    .replace(/[–−]/g, "-")
    .normalize("NFD").replace(/\p{M}/gu, "")
    .replace(/\$\s?(\d[\d,]*(?:\.\d+)?)/g, "$1 dollars")
    .replace(/(\d)\s?%/g, "$1 percent")
    .replace(/\b(\d+)(?:st|nd|rd|th)\b/gi, (_, d) => ordinalWords(d))
    .replace(/\d[\d,]*(?:\.\d+)?/g, numberToWords)
    .replace(/&/g, " and ").replace(/\+/g, " plus ").replace(/=/g, " equals ").replace(/@/g, " at ");
  s = [...s].filter((c) => /\s/.test(c) || (c in vocab && c.length === 1 && c !== "▁")).join("");
  s = s.replace(/\s+/g, " ").trim();
  if (!/\p{L}/u.test(s)) return { error: "読み上げられる英単語が含まれていません。" };
  if (!/[.!?]$/.test(s)) s += ".";
  return { text: s };
}

export function encodeText(text, vocab) {
  const eos = vocab["</s>"], space = vocab["▁"];
  const ids = [space];
  for (const c of text) ids.push(c === " " ? space : vocab[c]);
  ids.push(eos);
  return BigInt64Array.from(ids.map(BigInt));
}

export class Cancelled extends Error {}

export class SpeechT5 {
  constructor(ort, { encoder, decoder, vocoder }) {
    this.ort = ort;
    this.encoder = encoder;
    this.decoder = decoder;
    this.vocoder = vocoder;
    this.pastNames = decoder.inputNames.filter((n) => n.startsWith("past_key_values"));
  }

  async encode(ids) {
    const input = new this.ort.Tensor("int64", ids, [1, ids.length]);
    const r = await this.encoder.run({ input_ids: input });
    return { hidden: r.encoder_outputs, mask: r.encoder_attention_mask, length: ids.length };
  }

  // transformers の generate_speech と同じ停止条件 (threshold=0.5, maxlenratio=20)。
  // 止まらない場合に備え、文字数に応じた上限も設ける
  async decode(encoded, speaker, { onStep, isCancelled } = {}) {
    const { ort } = this;
    const speakerTensor = new ort.Tensor("float32", speaker, [1, 512]);
    const emptyPast = new ort.Tensor("float32", new Float32Array(0), [1, 12, 0, 64]);
    const maxSteps = Math.min(Math.floor((encoded.length / 2) * 20), encoded.length * 6 + 100);
    let sequence = new ort.Tensor("float32", new Float32Array(80), [1, 1, 80]);
    let past = null;
    const parts = [];
    for (let step = 1; ; step++) {
      if (isCancelled?.()) throw new Cancelled();
      const feeds = {
        speaker_embeddings: speakerTensor,
        encoder_hidden_states: encoded.hidden,
        encoder_attention_mask: encoded.mask,
        output_sequence: sequence,
        use_cache_branch: new ort.Tensor("bool", [past !== null], [1]),
      };
      for (const name of this.pastNames) feeds[name] = past ? past[name] : emptyPast;
      const r = await this.decoder.run(feeds);
      // use_cache 分岐では encoder 側の key/value は更新されない
      const next = {};
      for (const name of this.pastNames) {
        next[name] = past && name.includes(".encoder.") ? past[name] : r[name.replace("past_key_values", "present")];
      }
      past = next;
      sequence = r.output_sequence_out;
      parts.push(r.spectrum.data);
      onStep?.(step, maxSteps);
      if (r.prob.data.some((p) => p >= 0.5) || step >= maxSteps) break;
      await new Promise((resolve) => setTimeout(resolve));
    }
    const total = parts.reduce((n, p) => n + p.length, 0);
    const spectrogram = new Float32Array(total);
    let offset = 0;
    for (const p of parts) {
      spectrogram.set(p, offset);
      offset += p.length;
    }
    const { waveform } = await this.vocoder.run({
      spectrogram: new ort.Tensor("float32", spectrogram, [total / 80, 80]),
    });
    return Float32Array.from(waveform.data);
  }
}
