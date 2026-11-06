import { Engine } from "./engine.js";
import { MicError, MicRecorder, micAvailable, prepareRecording, wavUrl } from "./recorder.js";
import { attachHeatTooltip, drawHeatmap, drawWaveform } from "./charts.js";
import { MAX_INPUT_CHARS } from "./tts.js";

const SAMPLE_RATE = 16000;
const HOP = 256;
const MAX_SEC = 8;
const RECORD_LIMIT_SEC = 12;
const TARGET_NAMES = { female: "女性話者", male: "男性話者" };

const $ = (id) => document.getElementById(id);
const fmt = (v) => v.toFixed(3);
const engine = new Engine();

const state = { wave: null, attack: null, urls: new Map(), busy: false };

function setUrl(audioId, samples) {
  const old = state.urls.get(audioId);
  if (old) URL.revokeObjectURL(old);
  const url = wavUrl(samples);
  state.urls.set(audioId, url);
  $(audioId).src = url;
}

function clearUrl(audioId) {
  const old = state.urls.get(audioId);
  if (old) URL.revokeObjectURL(old);
  state.urls.delete(audioId);
  $(audioId).removeAttribute("src");
  $(audioId).load();
}

function showError(id, message) {
  const node = $(id);
  node.textContent = message || "";
  node.hidden = !message;
}

function setBusy(busy) {
  state.busy = busy;
  $("record-start").disabled = busy;
  $("attack-start").disabled = busy || !state.wave;
  $("clone-start").disabled = busy || !state.attack;
  $("clone-text").disabled = busy;
  document.querySelectorAll('input[name="steps"]').forEach((r) => (r.disabled = busy));
  document.querySelectorAll('input[name="target"]').forEach((r) => (r.disabled = busy));
}

function unlock(stepId, open) {
  $(stepId).classList.toggle("is-locked", !open);
}

function setProgress(barId, labelId, fraction, label) {
  $(barId).style.width = `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`;
  $(labelId).textContent = label;
}

function describeTime(sec) {
  if (sec < 60) return `約 ${Math.max(1, Math.round(sec))} 秒`;
  return `約 ${Math.round(sec / 60)} 分`;
}

// ---- Step 1: 録音 ----------------------------------------------------------

let recorder = null;
let liveFrames = [];
let stopRequested = false;

function resetAfterRecording() {
  state.attack = null;
  $("attack-result").hidden = true;
  $("attack-progress").hidden = true;
  showError("attack-error");
  resetClone();
  unlock("step-clone", false);
}

function resetClone() {
  $("clone-result").hidden = true;
  $("clone-progress").hidden = true;
  $("clone-summary").hidden = true;
  showError("clone-error");
  for (const v of ["clean", "protected"]) {
    clearUrl(`clone-audio-${v}`);
    $(`clone-verdict-${v}`).replaceChildren();
  }
}

async function startRecording() {
  showError("record-error");
  stopRequested = false;
  setBusy(true);
  liveFrames = [];
  try {
    recorder = new MicRecorder();
    const canvas = $("record-wave");
    canvas.hidden = false;
    let lastDraw = 0;
    await recorder.start((frame, elapsed) => {
      liveFrames.push(frame);
      const now = performance.now();
      if (now - lastDraw > 60) {
        lastDraw = now;
        $("record-timer").textContent = `${elapsed.toFixed(1)} 秒`;
        const recent = liveFrames.slice(-Math.ceil((SAMPLE_RATE * 4) / 128));
        const merged = new Float32Array(recent.reduce((n, f) => n + f.length, 0));
        let o = 0;
        for (const f of recent) {
          merged.set(f, o);
          o += f.length;
        }
        drawWaveform(canvas, merged, "#31302e");
      }
      if (elapsed >= RECORD_LIMIT_SEC && !stopRequested) stopRecording();
    });
    $("record-start").hidden = true;
    $("record-stop").hidden = false;
    $("record-timer").hidden = false;
    $("record-timer").textContent = "0.0 秒";
  } catch (e) {
    recorder = null;
    $("record-wave").hidden = true;
    setBusy(false);
    showError("record-error", e instanceof MicError ? e.message : `録音を開始できませんでした: ${e.message}`);
  }
}

async function stopRecording() {
  if (stopRequested || !recorder) return;
  stopRequested = true;
  const active = recorder;
  recorder = null;
  $("record-stop").hidden = true;
  $("record-start").hidden = false;
  $("record-timer").hidden = true;
  try {
    const captured = await active.stop();
    const wave = await prepareRecording(captured, MAX_SEC);
    state.wave = wave;
    resetAfterRecording();
    setUrl("record-audio", wave);
    setUrl("compare-clean", wave);
    $("record-result").hidden = false;
    drawWaveform($("record-wave"), wave, "#31302e");
    $("record-start").lastChild.textContent = "録音し直す";
    unlock("step-attack", true);
    engine.call("init").catch(() => {});
  } catch (e) {
    $("record-wave").hidden = true;
    showError("record-error", e instanceof MicError ? e.message : `録音の処理に失敗しました: ${e.message}`);
  } finally {
    setBusy(false);
  }
}

// ---- Step 2: 敵対的音声 ----------------------------------------------------

function stat(dl, label, value) {
  const box = document.createElement("div");
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.textContent = value;
  box.append(dt, dd);
  dl.appendChild(box);
}

function range(values, lowPct, highPct) {
  const sorted = Float32Array.from(values).sort();
  return [sorted[Math.floor(sorted.length * lowPct)], sorted[Math.floor(sorted.length * highPct)]];
}

function renderAttackResult(r) {
  const dl = $("attack-stats");
  dl.replaceChildren();
  stat(dl, "マスクの二値性", `${Math.round(r.stats.binarity * 100)}%`);
  stat(dl, "SNR", `${r.snr.toFixed(1)} dB`);
  setUrl("compare-adv", r.protected);

  // 2枚のスペクトログラムは同じ範囲で塗り、差が見えるようにする
  const [lo, hi] = range(r.clean.values, 0.05, 0.998);
  drawHeatmap($("heat-clean"), r.clean, { lo, hi });
  drawHeatmap($("heat-adv"), r.adv, { lo, hi });
  drawHeatmap($("heat-mask"), r.mask, { lo: 0, hi: 1, invert: true });
  const db = { sampleRate: SAMPLE_RATE, hop: HOP, format: (v) => `${v.toFixed(1)} dB`, label: "パワー" };
  attachHeatTooltip($("heat-clean-figure"), $("heat-clean"), r.clean, db);
  attachHeatTooltip($("heat-adv-figure"), $("heat-adv"), r.adv, db);
  attachHeatTooltip($("heat-mask-figure"), $("heat-mask"), r.mask, {
    sampleRate: SAMPLE_RATE, hop: HOP, format: (v) => `m = ${v.toFixed(2)}`, label: "マスク",
  });
  $("attack-result").hidden = false;
}

async function startAttack() {
  showError("attack-error");
  resetClone();
  unlock("step-clone", false);
  state.attack = null;
  $("attack-result").hidden = true;
  const steps = +document.querySelector('input[name="steps"]:checked').value;
  const target = document.querySelector('input[name="target"]:checked').value;
  setBusy(true);
  $("attack-start").hidden = true;
  $("attack-cancel").hidden = false;
  $("attack-cancel").disabled = false;
  $("attack-progress").hidden = false;
  setProgress("attack-bar", "attack-status", 0, "モデルを準備しています...");
  try {
    const wave = state.wave.slice();
    const result = await engine.call("attack", { wave, steps, target }, (p) => {
      const remain = ((p.steps - p.step) * p.msPerStep) / 1000;
      setProgress("attack-bar", "attack-status", p.step / p.steps,
        `ステップ ${p.step} / ${p.steps}（残り ${describeTime(remain)}）`);
    });
    if (!result) {
      setProgress("attack-bar", "attack-status", 0, "中止しました。");
      return;
    }
    state.attack = result;
    setProgress("attack-bar", "attack-status", 1,
      `完了しました（${(result.msPerStep * steps / 1000).toFixed(0)} 秒、${result.msPerStep.toFixed(0)} ms/ステップ）。ターゲット: ${TARGET_NAMES[result.target.id]}`);
    renderAttackResult(result);
    unlock("step-clone", true);
    updateCloneInput();
  } catch (e) {
    showError("attack-error", `生成に失敗しました: ${e.message}`);
    $("attack-progress").hidden = true;
  } finally {
    $("attack-start").hidden = false;
    $("attack-cancel").hidden = true;
    setBusy(false);
  }
}

// ---- Step 3: クローン ------------------------------------------------------

let normalizeTimer = 0;

function updateCloneInput() {
  const text = $("clone-text").value;
  const counter = $("clone-counter");
  counter.textContent = `${text.length} / ${MAX_INPUT_CHARS}`;
  counter.classList.toggle("is-over", text.length > MAX_INPUT_CHARS);
  clearTimeout(normalizeTimer);
  normalizeTimer = setTimeout(async () => {
    const r = await engine.call("normalize", { text }).catch(() => null);
    if (!r || $("clone-text").value !== text) return;
    const hint = $("clone-normalized");
    if (r.error) hint.textContent = r.error;
    else if (r.text !== text.trim()) hint.textContent = `読み上げる文: ${r.text}`;
    else hint.textContent = "SpeechT5 は英語専用です。数字は英単語に展開して読み上げます。";
    if (!state.busy) $("clone-start").disabled = Boolean(r.error) || !state.attack;
  }, 150);
}

function setVerdict(id, cosSrc, cosTgt) {
  const node = $(id);
  node.replaceChildren();
  const nearTarget = cosTgt > cosSrc;
  const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  icon.setAttribute("class", "icon");
  icon.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `assets/icons.svg#${nearTarget ? "target" : "user-round"}`);
  icon.appendChild(use);
  const label = document.createElement("span");
  label.textContent = `${nearTarget ? "ターゲット話者に近い" : "本人に近い"}（本人 ${fmt(cosSrc)} / ターゲット ${fmt(cosTgt)}）`;
  node.append(icon, label);
}

function renderCloneResult(data) {
  const { clean, protected: prot } = data;
  const summary = $("clone-summary");
  if (prot.cosTgt > prot.cosSrc && clean.cosSrc > clean.cosTgt) {
    summary.textContent = `保護によりクローンの話者が変わりました。本人との類似度は ${fmt(clean.cosSrc)} から ${fmt(prot.cosSrc)} へ下がり、クローンはターゲット話者に近くなっています。`;
  } else if (prot.cosSrc < clean.cosSrc) {
    summary.textContent = `本人との類似度は ${fmt(clean.cosSrc)} から ${fmt(prot.cosSrc)} へ下がりましたが、話者が入れ替わるほどの差ではありません。反復回数を増やすか、録音し直すと結果が変わる場合があります。`;
  } else {
    summary.textContent = `今回は本人との類似度が下がりませんでした（${fmt(clean.cosSrc)} から ${fmt(prot.cosSrc)}）。SpeechT5 の合成結果は毎回変わるため、もう一度生成してみてください。`;
  }
  summary.hidden = false;
}

async function startClone() {
  showError("clone-error");
  resetClone();
  setBusy(true);
  $("clone-start").hidden = true;
  $("clone-cancel").hidden = false;
  $("clone-cancel").disabled = false;
  $("clone-progress").hidden = false;
  setProgress("clone-bar", "clone-status", 0, "準備しています...");
  const names = { clean: "録音からのクローン", protected: "敵対的音声からのクローン" };
  const order = { clean: 0, protected: 1 };
  try {
    const data = await engine.call("clone", { text: $("clone-text").value }, (p) => {
      if (p.kind === "load" && p.phase === "download") {
        const verb = p.fromCache ? "保存済みのモデルを読み込んでいます" : "音声合成モデルを取得しています";
        setProgress("clone-bar", "clone-status", p.loadedMiB / p.totalMiB,
          `${verb}（${p.loadedMiB.toFixed(0)} / ${p.totalMiB.toFixed(0)} MiB）`);
      } else if (p.kind === "load") {
        $("clone-status").textContent = "音声合成モデルを準備しています...";
      } else if (p.kind === "encode") {
        setProgress("clone-bar", "clone-status", 0, "文章を処理しています...");
      } else if (p.kind === "decode") {
        // 停止位置は事前に分からないので、1文字あたりの典型的なステップ数から見積もる
        const f = Math.min(0.98, p.step / (p.chars * 2.3 + 8));
        setProgress("clone-bar", "clone-status", (order[p.variant] + f) / 2, `${names[p.variant]}を合成しています...`);
      } else if (p.kind === "variant") {
        setUrl(`clone-audio-${p.variant}`, p.wave);
        setVerdict(`clone-verdict-${p.variant}`, p.cosSrc, p.cosTgt);
        $("clone-result").hidden = false;
      }
    });
    if (!data) {
      setProgress("clone-bar", "clone-status", 0, "中止しました。");
      return;
    }
    setProgress("clone-bar", "clone-status", 1, `完了しました。読み上げた文: ${data.text}`);
    renderCloneResult(data);
  } catch (e) {
    showError("clone-error", `クローンの生成に失敗しました: ${e.message}`);
    $("clone-progress").hidden = true;
  } finally {
    $("clone-start").hidden = false;
    $("clone-cancel").hidden = true;
    setBusy(false);
    updateCloneInput();
  }
}

// ---- 初期化 ----------------------------------------------------------------

function drawHeroArt() {
  // 時間周波数マスクを模した装飾。一部のビンだけを抜いて「削る」様子を表す
  const host = $("hero-art");
  const tones = ["sky", "purple", "pink", "orange", "teal"];
  let seed = 20261106;
  const rnd = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
  for (let row = 0; row < 11; row++) {
    for (let col = 0; col < 16; col++) {
      const tile = document.createElement("span");
      const level = 0.5 + 0.5 * Math.sin(col * 0.7 + row * 0.5) * Math.cos(row * 0.9 - col * 0.25);
      const r = rnd();
      if (r < 0.2) tile.dataset.tone = "gone";
      else if (r < 0.2 + level * 0.22) tile.dataset.tone = tones[Math.floor(rnd() * tones.length)];
      host.appendChild(tile);
    }
  }
}

function init() {
  drawHeroArt();
  $("clone-counter").textContent = `${$("clone-text").value.length} / ${MAX_INPUT_CHARS}`;
  const notice = $("env-notice");
  if (!micAvailable()) {
    notice.textContent = "このブラウザまたは接続ではマイクを使用できません。HTTPS で開いた最新のブラウザをご利用ください。";
    notice.hidden = false;
    $("record-start").disabled = true;
  } else if (!window.crossOriginIsolated) {
    notice.textContent = "この配信環境ではマルチスレッドが使えないため、処理に時間がかかります。";
    notice.hidden = false;
  }
  $("record-start").addEventListener("click", startRecording);
  $("record-stop").addEventListener("click", stopRecording);
  $("attack-start").addEventListener("click", startAttack);
  $("attack-cancel").addEventListener("click", () => {
    $("attack-cancel").disabled = true;
    engine.cancel();
  });
  $("clone-start").addEventListener("click", startClone);
  $("clone-cancel").addEventListener("click", () => {
    $("clone-cancel").disabled = true;
    engine.cancel();
  });
  $("clone-text").addEventListener("input", updateCloneInput);
}

init();
