// マスク最適化。anticloning_mask/optimize.py の run_mask_attack と同じ損失・更新を行う。
//
//   L = L_speaker + w_dist * l_dist + w_percept * l_percept + w_beta * L_beta
//
// L_speaker とその勾配は gradFn (ONNX) が返す。Adam + 勾配ノルムクリップも torch と同じ式。

export function cosine(a, b) {
  let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return ab / Math.sqrt(aa * bb);
}

function gaussian(rng) {
  const u = 1 - rng(), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export class MaskAttack {
  // gradFn(feats: Float32Array, Tf) -> Promise<{ loss, emb: Float32Array, grad: Float32Array }>
  constructor({ wave, chain, cfg, gradFn, theta = null, rng = Math.random }) {
    this.wave = wave;
    this.chain = chain;
    this.cfg = cfg;
    this.gradFn = gradFn;
    this.spec = chain.stft(wave);
    const size = this.spec.T * this.spec.F;
    this.size = size;
    this.theta = new Float64Array(size);
    if (theta) this.theta.set(theta);
    else for (let i = 0; i < size; i++) this.theta[i] = cfg.initStd * gaussian(rng);
    // l_percept が使う正規化振幅
    this.magNorm = new Float64Array(size);
    let maxMag = 0;
    for (let i = 0; i < size; i++) {
      const mag = Math.hypot(this.spec.re[i], this.spec.im[i]);
      this.magNorm[i] = mag;
      if (mag > maxMag) maxMag = mag;
    }
    for (let i = 0; i < size; i++) this.magNorm[i] /= maxMag + 1e-10;

    this.mask = new Float64Array(size);
    this.dm = new Float64Array(size);
    this.grad = new Float64Array(size);
    this.adamM = new Float64Array(size);
    this.adamV = new Float64Array(size);
    this.stepCount = 0;
    this.xAdv = new Float64Array(chain.outputLength(this.spec));
    this.gx = new Float64Array(this.xAdv.length);
  }

  // 1反復。保護音声と最終マスクは更新前の値を残す (optimize.py と同じ)
  async step() {
    const { chain, cfg, spec, theta, mask, dm, grad, xAdv, gx, wave, size, magNorm } = this;
    for (let i = 0; i < size; i++) mask[i] = 1 / (1 + Math.exp(-theta[i]));
    chain.istft(spec, mask, xAdv);
    const fb = chain.fbank(xAdv);
    const { loss: taskLoss, emb, grad: gFeats } = await this.gradFn(fb.feats, fb.Tf);
    chain.fbankAdjoint(gFeats, fb, gx);

    // 音響損失 l_dist (波形MSE)
    const n = Math.min(xAdv.length, wave.length);
    let lDist = 0;
    for (let i = 0; i < n; i++) {
      const d = xAdv[i] - wave[i];
      lDist += d * d;
      gx[i] += (cfg.wDist * 2 * d) / n;
    }
    lDist /= n;
    chain.istftAdjoint(spec, gx, dm);

    // 音響損失 l_percept と Beta損失 L_beta。どちらもマスクに直接かかる
    const a1 = cfg.betaAlpha - 1, b1 = cfg.betaBeta - 1;
    let lPercept = 0, lBeta = 0, norm2 = 0;
    for (let i = 0; i < size; i++) {
      const m = Math.min(Math.max(mask[i], 1e-10), 1 - 1e-10);
      lPercept += magNorm[i] * (1 - mask[i]);
      lBeta -= a1 * Math.log(m) + b1 * Math.log(1 - m);
      const dBeta = -(a1 / m - b1 / (1 - m)) / size;
      const dPercept = -magNorm[i] / size;
      const g = (dm[i] + cfg.wPercept * dPercept + cfg.wBeta * dBeta) * mask[i] * (1 - mask[i]);
      grad[i] = g;
      norm2 += g * g;
    }
    lPercept /= size;
    lBeta /= size;
    const total = taskLoss + cfg.wBeta * lBeta + cfg.wDist * lDist + cfg.wPercept * lPercept;

    if (Number.isFinite(total)) this._adam(norm2);
    return { total, taskLoss, lDist, lPercept, lBeta, emb };
  }

  _adam(norm2) {
    const { cfg, theta, grad, adamM, adamV, size } = this;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const clip = Math.min(cfg.gradClip / (Math.sqrt(norm2) + 1e-6), 1);
    this.stepCount += 1;
    const c1 = 1 - Math.pow(b1, this.stepCount);
    const c2 = Math.sqrt(1 - Math.pow(b2, this.stepCount));
    for (let i = 0; i < size; i++) {
      const g = grad[i] * clip;
      adamM[i] = b1 * adamM[i] + (1 - b1) * g;
      adamV[i] = b2 * adamV[i] + (1 - b2) * g * g;
      theta[i] -= (cfg.lr * adamM[i]) / c1 / (Math.sqrt(adamV[i]) / c2 + eps);
    }
  }

  // 最後の step() 時点のマスクの統計
  maskStats() {
    const { mask, size } = this;
    let sum = 0, binary = 0;
    for (let i = 0; i < size; i++) {
      sum += mask[i];
      if (mask[i] < 0.1 || mask[i] > 0.9) binary += 1;
    }
    return { mean: sum / size, binarity: binary / size };
  }

  snrDb() {
    const n = Math.min(this.xAdv.length, this.wave.length);
    let signal = 0, noise = 0;
    for (let i = 0; i < n; i++) {
      const d = this.wave[i] - this.xAdv[i];
      signal += this.wave[i] * this.wave[i];
      noise += d * d;
    }
    return 10 * Math.log10(signal / (noise + 1e-10));
  }
}
