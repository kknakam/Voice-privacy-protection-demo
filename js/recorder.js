// マイク録音と 16 kHz モノラルへの変換・整形。音声はこのブラウザの外へ出ない。

const TARGET_RATE = 16000;
const MIN_SEC = 2;

export class MicError extends Error {}

export function micAvailable() {
  return Boolean(window.isSecureContext && navigator.mediaDevices?.getUserMedia && window.AudioWorkletNode);
}

export class MicRecorder {
  // onFrame(samples: Float32Array, elapsedSec)
  async start(onFrame) {
    try {
      // 声の特徴を変えないよう、ブラウザ側の音声処理は無効にする
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
    } catch (e) {
      if (e.name === "NotAllowedError") throw new MicError("マイクの使用が許可されませんでした。ブラウザの設定で許可してください。");
      if (e.name === "NotFoundError") throw new MicError("マイクが見つかりませんでした。接続を確認してください。");
      throw new MicError(`マイクを開始できませんでした（${e.name}）。`);
    }
    this.ctx = new AudioContext();
    await this.ctx.resume();
    await this.ctx.audioWorklet.addModule(new URL("./capture-worklet.js", import.meta.url));
    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "capture-processor", { numberOfOutputs: 0 });
    this.chunks = [];
    this.count = 0;
    this.node.port.onmessage = (e) => {
      this.chunks.push(e.data);
      this.count += e.data.length;
      onFrame?.(e.data, this.count / this.ctx.sampleRate);
    };
    this.source.connect(this.node);
  }

  async stop() {
    const rate = this.ctx.sampleRate;
    this.source.disconnect();
    this.node.port.onmessage = null;
    this.stream.getTracks().forEach((t) => t.stop());
    await this.ctx.close();
    const raw = new Float32Array(this.count);
    let offset = 0;
    for (const c of this.chunks) {
      raw.set(c, offset);
      offset += c.length;
    }
    this.chunks = [];
    return { raw, rate };
  }
}

async function resample(raw, rate) {
  if (rate === TARGET_RATE) return raw;
  const length = Math.ceil((raw.length * TARGET_RATE) / rate);
  const offline = new OfflineAudioContext(1, length, TARGET_RATE);
  const buffer = offline.createBuffer(1, raw.length, rate);
  buffer.copyToChannel(raw, 0);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0).slice();
}

// 16 kHz へ変換し、前後の無音を除いて最大長に収め、ピークを揃える
export async function prepareRecording({ raw, rate }, maxSec) {
  const x = await resample(raw, rate);
  if (x.length < TARGET_RATE * 0.5) throw new MicError("録音が短すぎます。もう一度録音してください。");
  let mean = 0;
  for (let i = 0; i < x.length; i++) mean += x[i];
  mean /= x.length;
  for (let i = 0; i < x.length; i++) x[i] -= mean;

  const frame = Math.round(TARGET_RATE * 0.025);
  const frames = Math.floor(x.length / frame);
  const rms = new Float64Array(frames);
  let top = 0;
  for (let f = 0; f < frames; f++) {
    let e = 0;
    for (let i = 0; i < frame; i++) e += x[f * frame + i] ** 2;
    rms[f] = Math.sqrt(e / frame);
    if (rms[f] > top) top = rms[f];
  }
  if (top < 0.003) throw new MicError("音声がほとんど入っていません。マイクの入力レベルを確認してください。");
  const threshold = Math.max(top * 0.03, 0.001);
  let first = 0, last = frames - 1;
  while (first < frames && rms[first] < threshold) first++;
  while (last > first && rms[last] < threshold) last--;
  const margin = Math.round(TARGET_RATE * 0.15);
  const start = Math.max(0, first * frame - margin);
  const end = Math.min(x.length, Math.min((last + 1) * frame + margin, start + Math.floor(TARGET_RATE * maxSec)));
  const wave = x.slice(start, end);
  if (wave.length < TARGET_RATE * MIN_SEC) {
    throw new MicError(`発話が短すぎます。${MIN_SEC} 秒以上話してください。`);
  }
  let peak = 0;
  for (let i = 0; i < wave.length; i++) peak = Math.max(peak, Math.abs(wave[i]));
  const gain = 0.8 / peak;
  for (let i = 0; i < wave.length; i++) wave[i] *= gain;
  return wave;
}

// 再生用の 16 bit PCM WAV
export function wavUrl(samples, sampleRate = TARGET_RATE) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset, s) => [...s].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, v < 0 ? v * 0x8000 : v * 0x7fff, true);
  }
  return URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
}
