// マスク最適化に必要な信号処理の順伝播と随伴 (逆伝播)。
//
//   mask --> X' = X * mask --> x' = ISTFT(X') --> fbank(x') --> [ONNX: x-vector + 話者損失]
//
// ONNX が返す dL/d(特徴量) を、ここで dL/d(mask) まで戻す。すべて FFT で計算でき、
// 演算は torch.stft / torch.istft / speechbrain の Fbank + 文単位CMN と一致させてある。

class ComplexFFT {
  constructor(n) {
    this.n = n;
    this.cos = new Float64Array(n);
    this.sin = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      this.cos[k] = Math.cos((2 * Math.PI * k) / n);
      this.sin[k] = -Math.sin((2 * Math.PI * k) / n);
    }
    this.radices = [];
    let rest = n;
    for (const r of [4, 2, 3, 5]) {
      while (rest % r === 0) {
        this.radices.push(r);
        rest /= r;
      }
    }
    if (rest !== 1) throw new Error(`未対応のFFT長です: ${n}`);
    this.tr = new Float64Array(n);
    this.ti = new Float64Array(n);
    this.ar = new Float64Array(5);
    this.ai = new Float64Array(5);
  }

  // Stockham 型の順変換 (正規化なし)。逆変換は re と im を入れ替えて呼ぶ。
  forward(re, im) {
    const N = this.n;
    const { cos, sin, ar, ai } = this;
    let xr = re, xi = im, yr = this.tr, yi = this.ti;
    let n = N, s = 1;
    for (const r of this.radices) {
      const m = n / r;
      if (r === 2) {
        for (let p = 0; p < m; p++) {
          const wr = cos[p * s], wi = sin[p * s];
          for (let q = 0; q < s; q++) {
            const i0 = q + s * p, i1 = i0 + s * m;
            const o0 = q + s * 2 * p, o1 = o0 + s;
            const dr = xr[i0] - xr[i1], di = xi[i0] - xi[i1];
            yr[o0] = xr[i0] + xr[i1];
            yi[o0] = xi[i0] + xi[i1];
            yr[o1] = dr * wr - di * wi;
            yi[o1] = dr * wi + di * wr;
          }
        }
      } else if (r === 4) {
        for (let p = 0; p < m; p++) {
          const w1r = cos[p * s], w1i = sin[p * s];
          const w2r = cos[2 * p * s], w2i = sin[2 * p * s];
          const w3r = cos[3 * p * s], w3i = sin[3 * p * s];
          for (let q = 0; q < s; q++) {
            const i0 = q + s * p, i1 = i0 + s * m, i2 = i1 + s * m, i3 = i2 + s * m;
            const o0 = q + s * 4 * p, o1 = o0 + s, o2 = o1 + s, o3 = o2 + s;
            const acr = xr[i0] + xr[i2], aci = xi[i0] + xi[i2];
            const amr = xr[i0] - xr[i2], ami = xi[i0] - xi[i2];
            const bdr = xr[i1] + xr[i3], bdi = xi[i1] + xi[i3];
            const bmr = xr[i1] - xr[i3], bmi = xi[i1] - xi[i3];
            yr[o0] = acr + bdr;
            yi[o0] = aci + bdi;
            const a1r = amr + bmi, a1i = ami - bmr;
            yr[o1] = a1r * w1r - a1i * w1i;
            yi[o1] = a1r * w1i + a1i * w1r;
            const a2r = acr - bdr, a2i = aci - bdi;
            yr[o2] = a2r * w2r - a2i * w2i;
            yi[o2] = a2r * w2i + a2i * w2r;
            const a3r = amr - bmi, a3i = ami + bmr;
            yr[o3] = a3r * w3r - a3i * w3i;
            yi[o3] = a3r * w3i + a3i * w3r;
          }
        }
      } else {
        const step = N / r;
        for (let p = 0; p < m; p++) {
          for (let q = 0; q < s; q++) {
            for (let j = 0; j < r; j++) {
              const i = q + s * (p + j * m);
              ar[j] = xr[i];
              ai[j] = xi[i];
            }
            for (let k = 0; k < r; k++) {
              let sr = 0, si = 0;
              for (let j = 0; j < r; j++) {
                const t = (j * k * step) % N;
                sr += ar[j] * cos[t] - ai[j] * sin[t];
                si += ar[j] * sin[t] + ai[j] * cos[t];
              }
              const t = (p * k * s) % N;
              const o = q + s * (r * p + k);
              yr[o] = sr * cos[t] - si * sin[t];
              yi[o] = sr * sin[t] + si * cos[t];
            }
          }
        }
      }
      [xr, yr] = [yr, xr];
      [xi, yi] = [yi, xi];
      n = m;
      s *= r;
    }
    if (xr !== re) {
      re.set(xr);
      im.set(xi);
    }
  }
}

// 長さ n (偶数) の実数FFT。n/2 点の複素FFTで計算する。
class RealFFT {
  constructor(n) {
    this.n = n;
    this.h = n / 2;
    this.fft = new ComplexFFT(this.h);
    this.zr = new Float64Array(this.h);
    this.zi = new Float64Array(this.h);
    this.cos = new Float64Array(this.h + 1);
    this.sin = new Float64Array(this.h + 1);
    for (let k = 0; k <= this.h; k++) {
      this.cos[k] = Math.cos((2 * Math.PI * k) / n);
      this.sin[k] = Math.sin((2 * Math.PI * k) / n);
    }
  }

  // x[n] -> re/im[off .. off + n/2]
  forward(x, re, im, off = 0) {
    const { h, zr, zi, cos, sin } = this;
    for (let i = 0; i < h; i++) {
      zr[i] = x[2 * i];
      zi[i] = x[2 * i + 1];
    }
    this.fft.forward(zr, zi);
    for (let k = 0; k <= h; k++) {
      const a = k % h, b = (h - k) % h;
      const er = (zr[a] + zr[b]) / 2, ei = (zi[a] - zi[b]) / 2;
      const dr = (zr[a] - zr[b]) / 2, di = (zi[a] + zi[b]) / 2;
      // O = -i * (dr + i*di) = di - i*dr, X = E + (c - i*s) * O
      const c = cos[k], s = sin[k];
      re[off + k] = er + c * di - s * dr;
      im[off + k] = ei - c * dr - s * di;
    }
    im[off] = 0;
    im[off + h] = 0;
  }

  // re/im[off .. off + n/2] -> x[n] (torch.fft.irfft と同じ 1/n 正規化)
  inverse(re, im, x, off = 0) {
    const { h, zr, zi, cos, sin } = this;
    for (let k = 0; k < h; k++) {
      const xr = re[off + k], xi = k === 0 ? 0 : im[off + k];
      const yr = re[off + h - k], yi = k === 0 ? 0 : im[off + h - k];
      const er = (xr + yr) / 2, ei = (xi - yi) / 2;
      const dr = (xr - yr) / 2, di = (xi + yi) / 2;
      const c = cos[k], s = sin[k];
      const or = dr * c - di * s, oi = dr * s + di * c;
      zr[k] = er - oi;
      zi[k] = ei + or;
    }
    this.fft.forward(zi, zr);
    for (let i = 0; i < h; i++) {
      x[2 * i] = zr[i] / h;
      x[2 * i + 1] = zi[i] / h;
    }
  }
}

export class SignalChain {
  constructor(constants) {
    const { stft, fbank } = constants;
    this.N = stft.nFft;
    this.hop = stft.hop;
    this.F = this.N / 2 + 1;
    this.rfft = new RealFFT(this.N);
    this.frame = new Float64Array(this.N);
    // torch.hann_window (periodic)
    this.hann = new Float64Array(this.N);
    for (let i = 0; i < this.N; i++) this.hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / this.N);

    this.fbN = fbank.nFft;
    this.fbHop = fbank.hop;
    this.fbF = this.fbN / 2 + 1;
    this.fbRfft = new RealFFT(this.fbN);
    this.fbFrame = new Float64Array(this.fbN);
    this.topDb = fbank.topDb;
    this.amin = fbank.amin;
    // torch.hamming_window (periodic)
    this.hamming = new Float64Array(this.fbN);
    for (let i = 0; i < this.fbN; i++) this.hamming[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / this.fbN);
    this.nMels = fbank.matrix[0].length;
    // フィルタバンクは疎なので、メルごとに非ゼロの範囲だけを持つ
    this.mel = [];
    for (let m = 0; m < this.nMels; m++) {
      let lo = -1, hi = -1;
      for (let k = 0; k < this.fbF; k++) {
        if (fbank.matrix[k][m] !== 0) {
          if (lo < 0) lo = k;
          hi = k;
        }
      }
      const w = new Float64Array(hi - lo + 1);
      for (let k = lo; k <= hi; k++) w[k - lo] = fbank.matrix[k][m];
      this.mel.push({ lo, w });
    }
  }

  // torch.stft(center=True, pad_mode="reflect")
  stft(x) {
    const { N, hop, F, hann, frame } = this;
    const pad = N / 2;
    const L = x.length;
    const T = 1 + Math.floor(L / hop);
    const re = new Float64Array(T * F), im = new Float64Array(T * F);
    const at = (i) => {
      if (i < 0) return x[-i];
      if (i >= L) return x[2 * L - 2 - i];
      return x[i];
    };
    for (let t = 0; t < T; t++) {
      const start = t * hop - pad;
      for (let n = 0; n < N; n++) frame[n] = at(start + n) * hann[n];
      this.rfft.forward(frame, re, im, t * F);
    }
    return { re, im, F, T, L, env: this._istftEnvelope(T) };
  }

  _istftEnvelope(T) {
    const { N, hop, hann } = this;
    const env = new Float64Array(N + hop * (T - 1));
    for (let t = 0; t < T; t++) {
      for (let n = 0; n < N; n++) env[t * hop + n] += hann[n] * hann[n];
    }
    return env;
  }

  outputLength(spec) {
    return this.hop * (spec.T - 1);
  }

  // torch.istft(center=True)。mask は [T * F]、out は長さ hop * (T - 1)
  istft(spec, mask, out) {
    const { N, hop, F, hann, frame } = this;
    const { T, re, im, env } = spec;
    const pad = N / 2;
    const y = this._ola || (this._ola = new Float64Array(env.length));
    if (y.length !== env.length) this._ola = new Float64Array(env.length);
    const ola = this._ola;
    ola.fill(0);
    const mr = this._mr || (this._mr = new Float64Array(F));
    const mi = this._mi || (this._mi = new Float64Array(F));
    for (let t = 0; t < T; t++) {
      const off = t * F;
      for (let k = 0; k < F; k++) {
        mr[k] = re[off + k] * mask[off + k];
        mi[k] = im[off + k] * mask[off + k];
      }
      this.rfft.inverse(mr, mi, frame);
      const o = t * hop;
      for (let n = 0; n < N; n++) ola[o + n] += frame[n] * hann[n];
    }
    for (let i = 0; i < out.length; i++) out[i] = ola[i + pad] / env[i + pad];
    return out;
  }

  // dL/dx' -> dL/dmask。istft の随伴
  istftAdjoint(spec, gx, dm) {
    const { N, hop, F, hann, frame } = this;
    const { T, re, im, env } = spec;
    const pad = N / 2;
    const adj = this._adj && this._adj.length === env.length ? this._adj : (this._adj = new Float64Array(env.length));
    adj.fill(0);
    for (let i = 0; i < gx.length; i++) adj[i + pad] = gx[i] / env[i + pad];
    const gr = this._gr || (this._gr = new Float64Array(F));
    const gi = this._gi || (this._gi = new Float64Array(F));
    const last = F - 1;
    for (let t = 0; t < T; t++) {
      const o = t * hop;
      for (let n = 0; n < N; n++) frame[n] = adj[o + n] * hann[n];
      this.rfft.forward(frame, gr, gi);
      const off = t * F;
      for (let k = 0; k < F; k++) {
        const c = k === 0 || k === last ? 1 / N : 2 / N;
        dm[off + k] = c * (re[off + k] * gr[k] + im[off + k] * gi[k]);
      }
    }
    return dm;
  }

  // speechbrain Fbank (n_mels=24) + InputNormalization(sentence, std_norm=False)。
  // 返り値の feats は [Tf * 24]、state は fbankAdjoint に渡す
  fbank(wave, length = wave.length) {
    const { fbN, fbHop, fbF, nMels, hamming, fbFrame, mel, amin, topDb } = this;
    const pad = fbN / 2;
    const Tf = 1 + Math.floor(length / fbHop);
    const re = new Float64Array(Tf * fbF), im = new Float64Array(Tf * fbF);
    const melValues = new Float64Array(Tf * nMels);
    const db = new Float64Array(Tf * nMels);
    const power = new Float64Array(fbF);
    let dbMax = -Infinity, argMax = 0;
    for (let t = 0; t < Tf; t++) {
      const start = t * fbHop - pad;
      for (let n = 0; n < fbN; n++) {
        const i = start + n;
        fbFrame[n] = i >= 0 && i < length ? wave[i] * hamming[n] : 0;
      }
      const off = t * fbF;
      this.fbRfft.forward(fbFrame, re, im, off);
      for (let k = 0; k < fbF; k++) power[k] = re[off + k] * re[off + k] + im[off + k] * im[off + k];
      for (let m = 0; m < nMels; m++) {
        const { lo, w } = mel[m];
        let v = 0;
        for (let j = 0; j < w.length; j++) v += power[lo + j] * w[j];
        const idx = t * nMels + m;
        melValues[idx] = v;
        const d = 10 * Math.log10(Math.max(v, amin));
        db[idx] = d;
        if (d > dbMax) {
          dbMax = d;
          argMax = idx;
        }
      }
    }
    const floor = dbMax - topDb;
    const clipped = new Uint8Array(Tf * nMels);
    const mean = new Float64Array(nMels);
    for (let i = 0; i < db.length; i++) {
      if (db[i] < floor) {
        db[i] = floor;
        clipped[i] = 1;
      }
      mean[i % nMels] += db[i];
    }
    for (let m = 0; m < nMels; m++) mean[m] /= Tf;
    const feats = new Float32Array(Tf * nMels);
    for (let i = 0; i < db.length; i++) feats[i] = db[i] - mean[i % nMels];
    return { feats, Tf, state: { re, im, melValues, clipped, argMax, length } };
  }

  // dL/d特徴量 -> dL/dx'。CMN の平均は PyTorch 実装でも勾配を切っている (.data)
  fbankAdjoint(gFeats, fb, out) {
    const { fbN, fbHop, fbF, nMels, hamming, fbFrame, mel, amin } = this;
    const { re, im, melValues, clipped, argMax, length } = fb.state;
    const Tf = fb.Tf;
    const pad = fbN / 2;
    const toDb = 10 / Math.LN10;
    const gDb = new Float64Array(gFeats.length);
    let gMax = 0;
    for (let i = 0; i < gFeats.length; i++) {
      // 下限で切られたビンの勾配は、下限を決めている最大ビン (amax) へ流れる
      if (clipped[i]) gMax += gFeats[i];
      else gDb[i] = gFeats[i];
    }
    gDb[argMax] += gMax;
    out.fill(0);
    const gPower = new Float64Array(fbF);
    const zr = new Float64Array(fbF), zi = new Float64Array(fbF);
    for (let t = 0; t < Tf; t++) {
      gPower.fill(0);
      for (let m = 0; m < nMels; m++) {
        const idx = t * nMels + m;
        if (melValues[idx] <= amin || gDb[idx] === 0) continue;
        const gMel = (gDb[idx] * toDb) / melValues[idx];
        const { lo, w } = mel[m];
        for (let j = 0; j < w.length; j++) gPower[lo + j] += gMel * w[j];
      }
      const off = t * fbF;
      for (let k = 0; k < fbF; k++) {
        const c = k === 0 || k === fbF - 1 ? 2 : 1;
        zr[k] = c * gPower[k] * re[off + k];
        zi[k] = c * gPower[k] * im[off + k];
      }
      this.fbRfft.inverse(zr, zi, fbFrame);
      const start = t * fbHop - pad;
      for (let n = 0; n < fbN; n++) {
        const i = start + n;
        if (i >= 0 && i < length) out[i] += fbN * fbFrame[n] * hamming[n];
      }
    }
    return out;
  }

  // 表示用の dB スペクトログラム [T * F]
  logMagnitude(wave) {
    const spec = this.stft(wave);
    const out = new Float32Array(spec.T * spec.F);
    for (let i = 0; i < out.length; i++) {
      out[i] = 10 * Math.log10(spec.re[i] * spec.re[i] + spec.im[i] * spec.im[i] + 1e-10);
    }
    return { values: out, F: spec.F, T: spec.T };
  }
}
