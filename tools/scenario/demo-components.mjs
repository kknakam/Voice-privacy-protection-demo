// 「デモ」のタブ (content/demo.mdx) で使うコンポーネント。HTML は、site/css/style.css のクラスだけで書く (style 属性は使わない)。
//
// 文章 (見出し、説明、ボタンの文字、補足) は、MDX で自由に書き換えられる。
// 一方、ボタン・進み具合・結果の置き場などは、site/js/demo/app.js が、id で探して動かす。
// そのため、id は、勝手に変えない (変えると、ビルドが、app.js が探す id が無いと知らせる)。
import { cloneElement, h } from "preact";
import { Icon, claimId, kids } from "./components.mjs";

const COLORS = new Set(["purple", "teal", "orange", "gray"]);   // 紫 = 本人、青緑 = 保護、橙 = 攻撃者、灰 = 中立
const need = (name, props, keys) => {
  for (const k of keys) if (props[k] === undefined || props[k] === "") throw new Error(`<${name}> には ${k} が要ります`);
};
const flag = (cond, attrs) => (cond ? attrs : {});

// ---- ステップ ------------------------------------------------------------------------------

// 1 つのステップ (見出しと、中身)。id は、URL のハッシュ (#step-record など) の行き先。locked は、前のステップが終わるまで閉じておく
export function DemoStep({ id, no, title, icon, color, locked, children }) {
  need("DemoStep", { id, no, title, icon }, ["id", "no", "title", "icon"]);
  if (!COLORS.has(color)) throw new Error(`<DemoStep> の color「${color}」は使えません (${[...COLORS].join(" / ")})`);
  claimId(id);
  return h(
    "article",
    { class: locked ? "step is-locked" : "step", id },
    h(
      "header",
      { class: "step-head" },
      h("span", { class: `sticker sticker-${color}` }, h(Icon, { name: icon })),
      h("div", null, h("span", { class: "eyebrow" }, `Step ${no}`), h("h3", null, title)),
    ),
    children,
  );
}

// ボタンの並び
export function Actions({ children }) {
  return h("div", { class: "actions" }, children);
}

// ボタン。secondary は、白いボタン。hidden は、最初は隠す (JS が出す)。disabled は、最初は押せない
export function Button({ id, icon, secondary, hidden, disabled, children }) {
  need("Button", { id, icon }, ["id", "icon"]);
  claimId(id);
  return h(
    "button",
    { class: secondary ? "btn btn-secondary" : "btn btn-primary", id, type: "button", ...flag(hidden, { hidden: true }), ...flag(disabled, { disabled: true }) },
    h(Icon, { name: icon }),
    children,
  );
}

// 録音の経過時間 (JS が、数字を書き換える)
export function Timer({ id, children }) {
  claimId(id);
  return h("span", { class: "timer", id, hidden: true, "aria-live": "off" }, children);
}

// 注意 (最初は隠す。JS が、文を入れて出す)
export function EnvNotice({ id = "env-notice" }) {
  claimId(id);
  return h("p", { class: "notice", id, hidden: true });
}

// エラーの文 (最初は隠す。JS が、文を入れて出す)
export function Alert({ id }) {
  claimId(id);
  return h("p", { class: "error", id, role: "alert", hidden: true });
}

// 録音中の波形 (JS が描く)
export function Waveform({ id, label }) {
  need("Waveform", { id, label }, ["id", "label"]);
  claimId(id);
  return h("canvas", { class: "waveform", id, hidden: true, "aria-label": label, role: "img" });
}

// 進み具合の棒。id は、"attack" なら、attack-progress・attack-bar・attack-status になる
export function Progress({ id }) {
  claimId(`${id}-progress`);
  claimId(`${id}-bar`);
  claimId(`${id}-status`);
  return h(
    "div",
    { class: "progress", id: `${id}-progress`, hidden: true },
    h("div", { class: "progress-track" }, h("div", { class: "progress-fill", id: `${id}-bar` })),
    h("p", { class: "progress-label", id: `${id}-status`, "aria-live": "polite" }),
  );
}

// 結果の置き場 (最初は隠す。JS が出す)
export function Result({ id, children }) {
  claimId(id);
  return h("div", { class: "result", id, hidden: true }, children);
}

// JS が、中身を入れる場所。tag は要素の種類、hidden は、最初は隠すか
export function Slot({ tag = "span", id, class: cls, hidden, children }) {
  claimId(id);
  return h(tag, { ...(cls ? { class: cls } : {}), id, ...flag(hidden, { hidden: true }) }, children);
}

// 統計の置き場 (JS が、項目を入れる)
export function Stats({ id }) {
  claimId(id);
  return h("dl", { class: "stats", id });
}

// ---- 音声 ------------------------------------------------------------------------------------

// 名前つきの再生器。ラベルの id は、既定で「<id>-label」。ラベルに、JS が書き込む場所 (<Slot>) を含めてもよい
export function AudioRow({ id, labelId, children }) {
  const label = labelId ?? `${id}-label`;
  claimId(id);
  claimId(label);
  return h(
    "div",
    { class: "audio-row" },
    h("span", { class: "audio-label", id: label }, children),
    h("audio", { id, controls: true, preload: "metadata", "aria-labelledby": label }),
  );
}

// 再生器を、2 つ並べる
export function AudioGrid({ children }) {
  return h("div", { class: "audio-grid" }, children);
}

// ---- 選択 ------------------------------------------------------------------------------------

// 選択肢の組 (ラジオボタン)。name は、JS が探す名前
export function Segmented({ id, name, legend, children }) {
  need("Segmented", { id, name, legend }, ["id", "name", "legend"]);
  claimId(id);
  const options = kids(children).map((c, i) => {
    if (c?.type !== Option) throw new Error(`<Segmented> の中に置けるのは、<Option> だけです (${i + 1} 番目)`);
    return cloneElement(c, { name });
  });
  return h("fieldset", { class: "segmented", id }, h("legend", null, legend), options);
}

// 選択肢 1 つ。value は、JS が読む値、note は、小さい補足、checked は、最初に選んでおく
export function Option({ name, value, checked, note, children }) {
  need("Option", { value }, ["value"]);
  return h(
    "label",
    null,
    h("input", { type: "radio", name, value, ...flag(checked, { checked: true }) }),
    h("span", null, children, note ? h("small", null, note) : null),
  );
}

// ---- スペクトログラム --------------------------------------------------------------------------

export function HeatGrid({ children }) {
  return h("div", { class: "heat-grid" }, children);
}

// スペクトログラム 1 枚 (JS が描く)。id が "heat-clean" なら、図の id は heat-clean-figure
export function Heat({ id, label, children }) {
  need("Heat", { id, label }, ["id", "label"]);
  claimId(id);
  claimId(`${id}-figure`);
  return h("figure", { class: "heat-figure", id: `${id}-figure` }, h("figcaption", null, children), h("canvas", { class: "heat", id, role: "img", "aria-label": label }));
}

// 色の目盛り。low と high は、両端の説明、note は、軸の説明
export function HeatScale({ low, high, children }) {
  need("HeatScale", { low, high }, ["low", "high"]);
  return h("div", { class: "heat-scale" }, h("span", null, low), h("span", { class: "heat-scale-bar" }), h("span", null, high), h("span", { class: "heat-scale-note" }, children));
}

// ---- 文章の入力 --------------------------------------------------------------------------------

// 文章の入力欄。label は見出し、value は、最初に入れておく文、counter は、文字数の表示の id、hint は、入力欄の下の説明 (JS が、書き換える)
export function TextField({ id, label, rows = 3, maxlength, lang, value, counterId, counter, hintId, children }) {
  need("TextField", { id, label, maxlength, counterId, counter, hintId }, ["id", "label", "maxlength", "counterId", "counter", "hintId"]);
  claimId(id);
  claimId(`${id}-label`);
  claimId(counterId);
  claimId(hintId);
  return h(
    "div",
    { class: "field" },
    h("div", { class: "field-head" }, h("label", { for: id, id: `${id}-label` }, label), h("span", { class: "counter", id: counterId }, counter)),
    h("textarea", { id, rows: String(rows), maxlength: String(maxlength), ...(lang ? { lang } : {}), spellcheck: "false" }, value),
    h("p", { class: "hint", id: hintId, "aria-live": "polite" }, children),
  );
}

// ---- クローンの結果 ------------------------------------------------------------------------------

export function CloneGrid({ children }) {
  return h("div", { class: "clone-grid" }, children);
}

// クローンの結果のカード 1 枚。name が "clean" なら、clone-card-clean・clone-audio-clean・clone-verdict-clean になる (JS が、この名前で探す)
export function CloneCard({ name: k, eyebrow, title, audioLabel }) {
  need("CloneCard", { name: k, eyebrow, title, audioLabel }, ["name", "eyebrow", "title", "audioLabel"]);
  claimId(`clone-card-${k}`);
  claimId(`clone-audio-${k}`);
  claimId(`clone-verdict-${k}`);
  return h(
    "section",
    { class: "clone-card", id: `clone-card-${k}` },
    h("span", { class: "eyebrow" }, eyebrow),
    h("h4", null, title),
    h("audio", { id: `clone-audio-${k}`, controls: true, preload: "metadata", "aria-label": audioLabel }),
    h("p", { class: "verdict", id: `clone-verdict-${k}` }),
  );
}

export const demoComponents = {
  DemoStep, Actions, Button, Timer, EnvNotice, Alert, Waveform, Progress, Result, Slot, Stats,
  AudioRow, AudioGrid, Segmented, Option, HeatGrid, Heat, HeatScale, TextField, CloneGrid, CloneCard,
};
