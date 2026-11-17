// 日本語に追加学習した SpeechT5。tts.js (英語) の ONNX とは入出力が異なるため、別のクラスにしている
//   ja_encoder.onnx         input_ids -> encoder_outputs, encoder_attention_mask, present.{i}.encoder.{key,value}
//   ja_decoder_step.onnx    KV キャッシュを入出力に持つ 1 ステップ分 (分岐なし)
//   ja_postnet_vocoder.onnx スペクトログラム -> postnet -> HiFi-GAN -> 波形

import { Cancelled } from "./tts.js";

// 文の終わりの検出: デコーダが出す align (選んだ cross-attention のヘッドの平均) の最大位置が、入力の末尾 END_TAIL 記号以内に
// 届いたら、その END_MARGIN ステップ後に止める。モデル自身の停止判定が働かず、同じ文を繰り返して読んでしまう場合 (JVS の声で
// 起きやすい) を防ぐ。余裕は、届いてからモデルが止まるまでの典型的な差 (約 4 ステップ) より長く取る。
// 評価 (30 文 x 8 話者): 崩れた生成 14 件の文字誤り率 100% -> 0%、正常な生成は 1.62% -> 1.59% (悪化した件数 0)
const END_TAIL = 2;
const END_MARGIN = 8;

const LAYERS = 6;
const HEADS = 12;
const HEAD_DIM = 64;

// 入力文を検証する。{ text, ids? } または { error } を返す。
//   定型文はあらかじめ計算した ID を返す (学習時と同じ音素化)。
//   それ以外は free が true のときだけ通し、ID はワーカーが OpenJTalk (WASM) で作る
export function normalizeJa(raw, ja, free = false) {
  const trimmed = raw.trim();
  if (!trimmed) return { error: "文章を入力してください。" };
  if (raw.length > ja.maxChars) return { error: `${ja.maxChars} 文字以内で入力してください。` };
  const key = trimmed.normalize("NFKC");
  const hit = ja.presets.find((p) => p.text.normalize("NFKC") === key);
  if (hit) return { text: hit.text, ids: hit.ids };
  if (free) return { text: trimmed };
  return { error: "現在は定型文のみ読み上げられます。候補から選んでください。" };
}

export class SpeechT5Ja {
  constructor(ort, { encoder, decoder, vocoder }) {
    this.ort = ort;
    this.encoder = encoder;
    this.decoder = decoder;
    this.vocoder = vocoder;
  }

  async encode(ids) {
    const input = new this.ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
    const r = await this.encoder.run({ input_ids: input });
    const cross = {};
    for (let i = 0; i < LAYERS; i++) {
      cross[`past_key_values.${i}.encoder.key`] = r[`present.${i}.encoder.key`];
      cross[`past_key_values.${i}.encoder.value`] = r[`present.${i}.encoder.value`];
    }
    return { mask: r.encoder_attention_mask, cross, length: ids.length };
  }

  // 停止条件: transformers の generate_speech と同じ (停止確率の和が 0.5 以上、上限は入力長の 5 倍) に、上の「文の終わりの検出」を加える
  async decode(encoded, speaker, { onStep, isCancelled, endGuard = true } = {}) {
    const { ort } = this;
    const speakerTensor = new ort.Tensor("float32", speaker, [1, 512]);
    const empty = () => new ort.Tensor("float32", new Float32Array(0), [1, HEADS, 0, HEAD_DIM]);
    const past = { ...encoded.cross };
    for (let i = 0; i < LAYERS; i++) {
      past[`past_key_values.${i}.decoder.key`] = empty();
      past[`past_key_values.${i}.decoder.value`] = empty();
    }
    const maxSteps = encoded.length * 5;
    let sequence = new ort.Tensor("float32", new Float32Array(80), [1, 1, 80]);
    const parts = [];
    let reachedEnd = 0; // 注意が入力の末尾に届いたステップ (届いていなければ 0)
    for (let step = 1; ; step++) {
      if (isCancelled?.()) throw new Cancelled();
      const r = await this.decoder.run({
        speaker_embeddings: speakerTensor,
        encoder_attention_mask: encoded.mask,
        output_sequence: sequence,
        ...past,
      });
      for (let i = 0; i < LAYERS; i++) {
        past[`past_key_values.${i}.decoder.key`] = r[`present.${i}.decoder.key`];
        past[`past_key_values.${i}.decoder.value`] = r[`present.${i}.decoder.value`];
      }
      sequence = r.output_sequence_out;
      parts.push(r.spectrum.data);
      onStep?.(step, maxSteps);
      let stop = 0;
      for (const p of r.prob.data) stop += p;
      if (endGuard && r.align && reachedEnd === 0) {
        const a = r.align.data;
        let best = 0;
        for (let k = 1; k < a.length; k++) if (a[k] > a[best]) best = k;
        if (best >= a.length - 1 - END_TAIL) reachedEnd = step;
      }
      if (stop >= 0.5 || step >= maxSteps || (reachedEnd > 0 && step - reachedEnd >= END_MARGIN)) break;
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
