import { Engine } from "./engine.js";
import { MicError, MicRecorder, micAvailable, prepareRecording, wavBlob, wavUrl } from "./recorder.js";
import * as api from "./api.js";
import { attachHeatTooltip, drawHeatmap, drawWaveform } from "./charts.js";
import { MAX_INPUT_CHARS } from "../tts/tts.js";

// 録音には、安全な接続 (HTTPS か localhost) と、AudioWorklet が要る。どちらかが無いと、録音の開始は押せない
const MIC_OK = micAvailable();
const SAMPLE_RATE = 16000;
const HOP = 256;
const MAX_SEC = 8;
const RECORD_LIMIT_SEC = 12;

const $ = (id) => document.getElementById(id);
const fmt = (v) => v.toFixed(3);
const engine = new Engine();

const state = { wave: null, attack: null, urls: new Map(), busy: false, ja: null, api: false, job: null, cancelRequested: false };

// 読み上げる言語は、入力した文章から決める。ひらがな・カタカナ・漢字が 1 字でもあれば日本語、なければ英語
const JAPANESE = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]/;
const detectLang = (text) => (JAPANESE.test(text) ? "ja" : "en");
const EN_HINT = "英語で読み上げます。数字は英単語に展開して読み上げます。";
const JA_HINT = "日本語で読み上げます（追加学習したモデル）。初回は、日本語の音素化の辞書も読み込みます。";
const JA_UNAVAILABLE = "この環境では、日本語の読み上げを用意していません。英語で入力してください。";
const JA_MAX_CHARS = 60;

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
  $("tab-demo-busy").hidden = !busy;   // 別のタブを見ている間も、デモの処理中だと分かるようにする
  $("record-start").disabled = busy || !MIC_OK;
  $("attack-start").disabled = busy || !state.wave || !state.api;
  $("clone-start").disabled = busy || !state.attack;
  $("clone-text").disabled = busy;
  document.querySelectorAll('input[name="steps"]').forEach((r) => (r.disabled = busy));
}

function unlock(stepId, open) {
  $(stepId).classList.toggle("is-locked", !open);
}

function setProgress(barId, labelId, fraction, label) {
  $(barId).style.width = `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`;
  $(labelId).textContent = label;
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

function stat(dl, label, value, note) {
  const box = document.createElement("div");
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.textContent = value;
  if (note) {
    const small = document.createElement("small");
    small.textContent = note;
    dd.appendChild(small);
  }
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
  const s = r.stats;
  stat(dl, "残した情報の割合", `${(s.maskMean * 100).toFixed(1)}%`, "マスクの平均値。大きいほど、元の音声の情報を多く残している");
  stat(dl, "SNR", `${s.snr.toFixed(1)} dB`, "大きいほど、元の音声からの変化が小さい");
  stat(dl, "UTMOS", `${s.utmosAdv.toFixed(2)} / 5.0`, `録音は ${s.utmosClean.toFixed(2)} / 5.0。大きいほど、音質がよい`);
  setUrl("compare-adv", r.protected);
  $("attack-target-name").textContent = `話者 ${r.target.id}`;
  $("attack-target-audio").src = `assets/targets/${r.target.id}.wav`;

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

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// 敵対的音声は、サーバで作る。録音を送り、順番待ちと進み具合を見て、結果を受け取る
async function startAttack() {
  showError("attack-error");
  resetClone();
  unlock("step-clone", false);
  state.attack = null;
  state.cancelRequested = false;
  $("attack-result").hidden = true;
  const steps = +document.querySelector('input[name="steps"]:checked').value;
  setBusy(true);
  $("attack-start").hidden = true;
  $("attack-cancel").hidden = false;
  $("attack-cancel").disabled = false;
  $("attack-progress").hidden = false;
  setProgress("attack-bar", "attack-status", 0, "録音をサーバへ送っています...");
  try {
    state.job = await api.submit(wavBlob(state.wave), steps);
    if (state.cancelRequested) api.cancel(state.job);
    for (;;) {
      await sleep(500);
      const s = await api.status(state.job);
      if (s.state === "queued") setProgress("attack-bar", "attack-status", 0, `順番待ちです（${s.position} 番目）`);
      else if (s.state === "running") setProgress("attack-bar", "attack-status", s.step / s.steps, s.step ? `ステップ ${s.step} / ${s.steps}` : `${s.message || "準備しています"}...`);
      else if (s.state === "cancelled") {
        setProgress("attack-bar", "attack-status", 0, "中止しました。");
        return;
      } else if (s.state === "error") throw new Error(s.message);
      else break;
    }
    const result = api.decode(await api.result(state.job));
    await engine.call("setSession", { eSrc: result.emb.src, eProtected: result.emb.adv, eTarget: result.emb.tgt });
    state.attack = result;
    setProgress("attack-bar", "attack-status", 1, `完了しました（${result.seconds.toFixed(0)} 秒）。ターゲット: 話者 ${result.target.id}`);
    renderAttackResult(result);
    unlock("step-clone", true);
    updateCloneInput();
  } catch (e) {
    showError("attack-error", `生成に失敗しました: ${e.message}`);
    $("attack-progress").hidden = true;
  } finally {
    state.job = null;
    $("attack-start").hidden = false;
    $("attack-cancel").hidden = true;
    setBusy(false);
  }
}

function cancelAttack() {
  $("attack-cancel").disabled = true;
  state.cancelRequested = true;
  if (state.job) api.cancel(state.job);
}

// ---- Step 3: クローン ------------------------------------------------------

let normalizeTimer = 0;

const currentLang = () => detectLang($("clone-text").value);
const maxChars = () => (currentLang() === "ja" ? JA_MAX_CHARS : MAX_INPUT_CHARS);

function updateCloneInput() {
  const text = $("clone-text").value;
  const lang = currentLang();
  $("clone-text").lang = lang;
  const counter = $("clone-counter");
  counter.textContent = `${text.length} / ${maxChars()}`;
  counter.classList.toggle("is-over", text.length > maxChars());
  clearTimeout(normalizeTimer);
  normalizeTimer = setTimeout(async () => {
    const hint = $("clone-normalized");
    if (lang === "ja" && state.ja && !state.ja.usable) {
      hint.textContent = JA_UNAVAILABLE;
      if (!state.busy) $("clone-start").disabled = true;
      return;
    }
    const r = await engine.call("normalize", { text, lang }).catch(() => null);
    if (!r || $("clone-text").value !== text) return;
    if (r.error) hint.textContent = r.error;
    else if (r.text !== text.trim()) hint.textContent = `読み上げる文: ${r.text}`;
    else hint.textContent = lang === "ja" ? JA_HINT : EN_HINT;
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
    const data = await engine.call("clone", { text: $("clone-text").value, lang: currentLang() }, (p) => {
      if (p.kind === "load" && p.phase === "download") {
        const verb = p.fromCache ? "保存済みのデータを読み込んでいます" : `${p.what ?? "音声合成モデル"}を取得しています`;
        setProgress("clone-bar", "clone-status", p.loadedMiB / p.totalMiB,
          `${verb}（${p.loadedMiB.toFixed(0)} / ${p.totalMiB.toFixed(0)} MiB）`);
      } else if (p.kind === "load") {
        $("clone-status").textContent = "音声合成モデルを準備しています...";
      } else if (p.kind === "encode") {
        setProgress("clone-bar", "clone-status", 0, "文章を処理しています...");
      } else if (p.kind === "decode") {
        // 停止位置は事前に分からないので、1文字あたりの典型的なステップ数から見積もる
        const f = Math.min(0.98, p.step / (p.expected ?? p.chars * 2.3 + 8));
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

// 日本語モデルと、自由入力のための辞書がそろっているときだけ、日本語の入力を受け付ける
async function setupJapanese() {
  const status = await engine.call("jaStatus").catch(() => null);
  state.ja = { usable: Boolean(status?.available && status.freeInput) };
  if (!state.ja.usable && currentLang() === "ja") updateCloneInput();   // 先に日本語を入力していたときだけ、案内を出す
}

function init() {
  drawHeroArt();
  $("clone-counter").textContent = `${$("clone-text").value.length} / ${MAX_INPUT_CHARS}`;
  setupJapanese();
  const notice = $("env-notice");
  if (!MIC_OK) {
    notice.textContent = window.isSecureContext
      ? "このブラウザは、録音に必要な機能 (AudioWorklet) に対応していません。最新の Chrome や Safari をご利用ください。"
      : `このページは、安全な接続ではないため、ブラウザがマイクの使用を止めています。https:// の URL か、http://localhost:ポート番号/ で開いてください（いまの URL: ${window.location.origin}）。`;
    notice.hidden = false;
    $("record-start").disabled = true;
  } else if (!window.crossOriginIsolated) {
    notice.textContent = "この配信環境ではマルチスレッドが使えないため、処理に時間がかかります。";
    notice.hidden = false;
  }
  api.available().then((ok) => {
    state.api = ok;
    if (!ok) {
      showError("attack-error", "この配信では、敵対的音声を作るサーバにつながっていないため、Step 2 を使えません。");
    }
    setBusy(state.busy);
  });
  $("record-start").addEventListener("click", startRecording);
  $("record-stop").addEventListener("click", stopRecording);
  $("attack-start").addEventListener("click", startAttack);
  $("attack-cancel").addEventListener("click", cancelAttack);
  $("clone-start").addEventListener("click", startClone);
  $("clone-cancel").addEventListener("click", () => {
    $("clone-cancel").disabled = true;
    engine.cancel();
  });
  $("clone-text").addEventListener("input", updateCloneInput);
  // 波形は表示幅を読んで描くので、別のタブで描いた分は、戻ったときに描き直す
  document.addEventListener("tabchange", (e) => {
    if (e.detail.tab === "demo" && state.wave && !recorder && !$("record-wave").hidden) {
      drawWaveform($("record-wave"), state.wave, "#31302e");
    }
  });
}

init();
