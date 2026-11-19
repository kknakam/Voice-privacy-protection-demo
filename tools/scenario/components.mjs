// 「研究のシナリオ」の MDX (content/scenario.mdx) で使うコンポーネント。
// HTML は、site/css/style.css のクラスだけで書く。CSP (style-src 'self') のため、style 属性は使わない。
// ビルド時 (Node) に、HTML の文字列へ変換するだけで、ブラウザには配信しない。
//
// 2 回に分けて描画する。1 回目で、図の番号と節の一覧を集め、2 回目で、それを使う (目次と、図への参照のため)。
import fs from "node:fs";
import path from "node:path";
import { Fragment, createContext, h, toChildArray } from "preact";
import { useContext } from "preact/hooks";

const COLORS = new Set(["purple", "teal", "orange", "gray"]);   // 紫 = 本人、青緑 = 保護、橙 = 攻撃者、灰 = 中立

const ILLUST_DIR = "assets/illust";                       // イラストを置く場所 (site/ からの相対)
const ILLUST_EXT = new Set([".svg", ".png", ".webp", ".jpg", ".jpeg"]);
const ILLUST_MAX_KIB = 300;

let icons = new Set();
let siteRoot = "";
let known = null;                                  // 1 回目に集めたもの (2 回目で使う)
let found = { figures: new Map(), blocks: [], ids: new Set() };    // この回に集めているもの
let figureNo = 0;

// 描画のたびに呼ぶ。icons は、site/assets/icons.svg にあるアイコンの id の集合。siteRoot は、公開するフォルダ (site/)。
// known は、前の回に collected() で得たもの
export function setup(options) {
  icons = options.icons;
  siteRoot = options.siteRoot ?? "";
  known = options.known ?? null;
  found = { figures: new Map(), blocks: [], ids: new Set() };
  figureNo = 0;
}
export const collected = () => found;

// 画面の部品の id を登録する。同じ id が 2 回出ると、止める (JS が、id で部品を探すため)
export function claimId(id) {
  if (!id) throw new Error("id が要ります");
  if (found.ids.has(id)) throw new Error(`id「${id}」が重複しています`);
  found.ids.add(id);
  return id;
}

const FigureContext = createContext({ caption: "" });

// 空白だけの文字列 (JSX の行の間の改行) を除いた、子要素の配列
export const kids = (children) => toChildArray(children).filter((c) => !(typeof c === "string" && c.trim() === ""));

export function Icon({ name, hidden = true, className = "icon" }) {
  if (!icons.has(name)) throw new Error(`アイコン「${name}」が、site/assets/icons.svg にありません`);
  return h("svg", { class: className, ...(hidden ? { "aria-hidden": "true" } : {}) }, h("use", { href: `assets/icons.svg#${name}` }));
}

// ---- ページ全体 ------------------------------------------------------------------------

// ページの見出しと、導入の 1 文
export function Title({ children }) {
  return h("h2", null, children);
}
export function Lead({ children }) {
  return h("p", { class: "section-lead" }, children);
}

// 目次。<Block toc="..."> を付けた節が、並んだ順に入る
export function Toc({ label }) {
  if (!label) throw new Error("<Toc> には label (目次の名前) が要ります");
  const items = (known?.blocks ?? []).filter((b) => b.toc);
  return h("nav", { class: "toc", "aria-label": label }, h("ol", null, items.map((b) => h("li", { key: b.id }, h("a", { href: `#${b.id}` }, b.toc)))));
}

// 節。id は、目次やリンクの行き先になる。見出しの id は「<id>-title」。toc は、目次に出す短い名前 (省くと、目次に出ない)
export function Block({ id, eyebrow, title, toc, class: cls, children }) {
  if (!id || !eyebrow || !title) throw new Error("<Block> には、id・eyebrow・title が要ります");
  if (found.blocks.some((b) => b.id === id)) throw new Error(`<Block id="${id}"> が重複しています`);
  found.blocks.push({ id, toc });
  return h(
    "section",
    { class: cls ? `block ${cls}` : "block", id, "aria-labelledby": `${id}-title` },
    h("div", null, h("span", { class: "eyebrow" }, eyebrow), h("h3", { id: `${id}-title` }, title)),
    children,
  );
}

// ---- 図 ----------------------------------------------------------------------------------

// 図。番号は、本文に出てくる順に自動で振る。id を付けると、<Ref to="id" /> で、本文から「図 N」と参照できる
export function Figure({ id, caption, children }) {
  if (!caption) throw new Error("<Figure> には caption が要ります");
  figureNo += 1;
  if (id) {
    if (found.figures.has(id)) throw new Error(`<Figure id="${id}"> が重複しています`);
    found.figures.set(id, figureNo);
  }
  return h(
    "figure",
    { class: "diagram" },
    h(FigureContext.Provider, { value: { caption } }, children),
    h("figcaption", null, "図 ", String(figureNo), "　", caption),
  );
}

// 図への参照 (「図 2」)。<Figure id="..."> の id を指定する
export function Ref({ to }) {
  if (known && !known.figures.has(to)) {
    throw new Error(`<Ref to="${to}" /> の図が、見つかりません (<Figure id="${to}"> があるか、確認してください)`);
  }
  return `図 ${known ? known.figures.get(to) : "?"}`;
}

// 手順の並び (ol)。間の矢印は自動で入る
function FlowList({ name, steps }) {
  return h(
    "ol",
    { class: "flow", "aria-label": name },
    steps.map((step, i) =>
      h(
        "li",
        { class: "flow-step", key: i },
        step,
        i < steps.length - 1
          ? h("span", { class: "flow-arrow", "aria-hidden": "true" }, h(Icon, { name: "arrow-right", hidden: false }))
          : null,
      ),
    ),
  );
}

// 流れの図。<Step> を並べると、間の矢印は自動で入る。label を省くと、図の caption が、読み上げ用の名前になる。
// loop に文字を渡すと、最後の手順の下から、最初の手順の下へ戻る矢印が付く (文字は、矢印の上に出る)
export function Flow({ label, loop, children }) {
  const { caption } = useContext(FigureContext);
  const steps = kids(children);
  if (steps.length === 0) throw new Error("<Flow> には、<Step> が 1 つ以上要ります");
  const name = label ?? caption;
  if (typeof name !== "string") throw new Error("caption に JSX を使った図の <Flow> には、label (文字列) が要ります");
  const list = h(FlowList, { name, steps });
  if (loop === undefined) return list;
  if (typeof loop !== "string" || !loop) throw new Error("<Flow> の loop には、矢印の上に出す文字 (文字列) を書きます");
  if (steps.length < 2 || steps.length > 8) throw new Error("<Flow loop> は、手順が 2〜8 個の流れで使えます");
  return h(
    "div",
    { class: "flow-wrap" },
    list,
    h("div", { class: `flow-return loop-n-${steps.length}` }, h("span", { class: "flow-return-label" }, h(Icon, { name: "rotate-ccw" }), loop)),
  );
}

// イラストの検査。置き場所は site/assets/illust/。image には、そこからの名前 (例: "mic.svg") を書く
function checkIllust(image, alt) {
  if (alt === undefined) throw new Error(`<Step image="${image}"> には alt が要ります (飾りだけの絵なら alt="" と書く)`);
  if (!ILLUST_EXT.has(path.extname(image).toLowerCase())) throw new Error(`イラスト「${image}」の形式は使えません (${[...ILLUST_EXT].join(" ")})`);
  if (/[\\/]/.test(image) || image.startsWith(".")) throw new Error(`image には、${ILLUST_DIR}/ の中のファイル名だけを書きます (「${image}」)`);
  const file = path.join(siteRoot, ILLUST_DIR, image);
  if (!fs.existsSync(file)) throw new Error(`イラスト「${image}」が、site/${ILLUST_DIR}/ にありません`);
  const kib = fs.statSync(file).size / 1024;
  if (kib > ILLUST_MAX_KIB) throw new Error(`イラスト「${image}」が大きすぎます (${kib.toFixed(0)} KiB。上限は ${ILLUST_MAX_KIB} KiB)`);
}

const RESULTS = { ok: ["circle-check", "outcome-ok"], ng: ["circle-x", "outcome-ng"] };   // 丸 (できた) と、ばつ (できない)

// デモのどの部分に当たるかを示すボタン (href があればリンク、なければ文字だけ)
function chipEl(demo, href, cls = "") {
  const extra = cls ? ` ${cls}` : "";
  return href
    ? h("a", { class: `demo-chip${extra}`, href }, demo, h(Icon, { name: "arrow-right" }))
    : h("span", { class: `demo-chip is-plain${extra}` }, demo);
}

// 1 手順を、カードとボタンに分けて作る (<Step> と <Branching> が使う)
function stepParts(props, cardClass = "", chipClass = "") {
  const { role, icon, color, title, demo, href, image, alt, result, resultLabel, children } = props;
  if (!COLORS.has(color)) throw new Error(`<Step> の color「${color}」は使えません (${[...COLORS].join(" / ")})`);
  if (!role || !icon || !title) throw new Error("<Step> には、role・icon・title が要ります");
  if (href && !demo) throw new Error("<Step> の href を付けるときは、demo (ボタンの文字) も付けます");
  if (result && !RESULTS[result]) throw new Error(`<Step> の result「${result}」は使えません (${Object.keys(RESULTS).join(" / ")})`);
  if (Boolean(result) !== Boolean(resultLabel)) throw new Error("<Step> の result と resultLabel は、両方付けるか、両方省きます");
  if (image) checkIllust(image, alt);
  const card = h(
    "div",
    { class: ["flow-card", image && "has-illust", result && `is-${result}`, cardClass].filter(Boolean).join(" ") },
    image
      ? h("span", { class: `flow-illust tint-${color}` }, h("img", { src: `${ILLUST_DIR}/${image}`, alt, loading: "lazy", decoding: "async" }))
      : h("span", { class: `sticker sticker-${color}` }, h(Icon, { name: icon })),
    h(
      "div",
      { class: "flow-body" },
      h("span", { class: "flow-role" }, role),
      h("h4", null, title),
      h("p", null, children),
      result ? h("div", { class: `outcome ${RESULTS[result][1]}` }, h(Icon, { name: RESULTS[result][0] }), resultLabel) : null,
    ),
  );
  return { card, chip: demo ? chipEl(demo, href, chipClass) : null };
}

// 流れの 1 手順。role は、役割の文字 (色だけに頼らないため)。本文は 1 行の文。
// demo を付けると、カードの下に、デモのどの部分に当たるかを示すボタンが付く (href を足すと、そこへのリンクになる)。
// image を付けると、アイコンの代わりにイラストが出る (alt は必須)。
// result="ok" (丸) か "ng" (ばつ) と resultLabel (文字) を付けると、目的が達成されたかどうかを、アイコンつきで示す
export function Step(props) {
  const { card, chip } = stepParts(props);
  return h(Fragment, null, card, chip);
}

// ---- 分岐のある流れ ----------------------------------------------------------------------

// 分岐の 1 段 (<Branching> の中だけで使う)。<Case> を 2 つ入れる (1 つ目が上の段、2 つ目が下の段)。
// demo と href を付けると、この段の下に、デモへのボタンが付く
export function Alts() {
  throw new Error("<Alts> は、<Branching> の中でだけ使えます");
}

// 分岐の 1 本。name は、枝の名前 (札になる)、color は、札の色。
// 中身が <Step> 1 つなら、その手順のカード。文だけなら、前後の手順を、その文を添えた 1 本の長い矢印でつなぐ (前後は、共通の <Step> にする)
export function Case() {
  throw new Error("<Case> は、<Alts> の中でだけ使えます");
}

function parseStage(v, i) {
  if (v?.type === Step) return { kind: "common", props: v.props };
  if (v?.type === Alts) {
    const cases = kids(v.props.children);
    if (cases.length !== 2 || cases.some((c) => c?.type !== Case)) throw new Error("<Alts> には、<Case> が 2 つ要ります (上の段と、下の段)");
    if (v.props.href && !v.props.demo) throw new Error("<Alts> の href を付けるときは、demo (ボタンの文字) も付けます");
    const parsed = cases.map((c) => {
      const { name, color, children } = c.props;
      if (!name) throw new Error("<Case> には name (枝の名前) が要ります");
      if (!COLORS.has(color)) throw new Error(`<Case name="${name}"> の color「${color}」は使えません (${[...COLORS].join(" / ")})`);
      const inner = kids(children);
      if (inner.length === 0) throw new Error(`<Case name="${name}"> の中身が空です`);
      const step = inner.length === 1 && inner[0]?.type === Step ? inner[0].props : null;
      if (step?.demo || step?.href) throw new Error(`<Case name="${name}"> の中の <Step> には、demo や href を付けられません (<Alts demo="..."> に付けます)`);
      return { name, color, step, text: children };
    });
    if (parsed.every((c) => !c.step)) throw new Error("<Alts> の <Case> が 2 つとも文だけです (少なくとも 1 つは <Step> を入れます)");
    return { kind: "alts", cases: parsed, demo: v.props.demo, href: v.props.href };
  }
  throw new Error(`<Branching> の中に置けるのは、<Step> と <Alts> だけです (${i + 1} 番目)`);
}

// 分岐のある流れ。<Step> (共通の手順。上下の段をまたぐ背の高いカード) と <Alts> (上下 2 本に分かれる手順) を、順に並べる。
// 広い画面では、共通の手順から 2 本の矢印が出て、次の共通の手順で 1 つに合わさる。狭い画面では、縦に積む
export function Branching({ label, children }) {
  if (typeof label !== "string" || !label) throw new Error("<Branching> には label (文字列) が要ります");
  const stages = kids(children).map(parseStage);
  const n = stages.length;
  if (n < 2 || n > 7) throw new Error("<Branching> は、<Step> と <Alts> が合わせて 2〜7 個の図で使えます");
  const arrowIcon = h(Icon, { name: "arrow-right", hidden: false });
  const arrow = (col, row) => h("span", { class: `flow-arrow h-only gc-${col} gr-${row}`, "aria-hidden": "true" }, arrowIcon);
  const caseEl = (c, col, row) => {
    const pill = h("span", { class: `case-pill pill-${c.color}` }, c.name);
    if (c.step) return h("div", { class: `case gc-${col} gr-${row}` }, pill, stepParts(c.step).card);
    return h("div", { class: `case is-plain gc-${col} gr-${row}` }, pill, h("p", { class: "plain-box" }, c.text));
  };
  const longArrow = (c, col, row) =>
    h(
      "span",
      { class: `flow-arrow is-skip h-only gc-${col} gs-3 gr-${row}` },
      h("span", { class: "skip-label" }, h("span", { class: `case-pill pill-${c.color}` }, c.name), h("span", null, c.text)),
      h("span", { class: "skip-line", "aria-hidden": "true" }),
      h(Icon, { name: "arrow-right" }),
    );
  const items = [];
  stages.forEach((st, k) => {
    const col = 2 * k + 1;
    const arrowCol = col + 1;
    const next = stages[k + 1];
    if (st.kind === "common") {
      const { card, chip } = stepParts(st.props);
      items.push(h("div", { class: `stage gc-${col} gr-12` }, card, chip));
    } else {
      st.cases.forEach((c) => {
        if (!c.step && (k === 0 || k === n - 1 || stages[k - 1].kind !== "common" || stages[k + 1].kind !== "common")) {
          throw new Error(`<Case name="${c.name}"> を文だけにできるのは、前後が <Step> のときだけです`);
        }
      });
      items.push(h("div", { class: "alts" }, st.cases.map((c, i) => caseEl(c, col, i + 1)), st.demo ? chipEl(st.demo, st.href, `gc-${col} gr-3`) : null));
    }
    if (!next) return;
    if (st.kind === "common" && next.kind === "common") items.push(arrow(arrowCol, 12));
    else if (st.kind === "common") next.cases.forEach((c, i) => items.push(c.step ? arrow(arrowCol, i + 1) : longArrow(c, arrowCol, i + 1)));
    else if (next.kind === "common") st.cases.forEach((c, i) => c.step && items.push(arrow(arrowCol, i + 1)));
    else throw new Error("<Alts> を続けて並べることはできません (間に <Step> を入れます)");
    items.push(h("span", { class: "flow-arrow v-only", "aria-hidden": "true" }, arrowIcon));   // 狭い画面の、縦の矢印
  });
  return h("div", { class: `branching stages-${n}`, role: "group", "aria-label": label }, items);
}

// 図 3 の格子の入れ物と、その凡例。格子の中身は、ブラウザで site/js/site/scenario.js が描く
export function MaskDemo({ label, strength = "成分の強さ", weak = "弱", strong = "強", keep = "残す", cut = "削る" }) {
  if (!label) throw new Error("<MaskDemo> には label (図の説明) が要ります");
  const cell = (cls) => h("span", { class: `cell ${cls}`, "aria-hidden": "true" });
  return h(
    Fragment,
    null,
    h("div", { class: "grid-demo", id: "mask-demo", role: "img", "aria-label": label }),
    h(
      "div",
      { class: "cells-key" },
      h("span", { class: "key-item" }, `${strength}（${weak}）`, [0, 1, 2, 3, 4].map((n) => cell(`cell-${n}`)), `（${strong}）`),
      h("span", { class: "key-item" }, cell("mask-keep"), keep),
      h("span", { class: "key-item" }, cell("mask-cut"), cut),
    ),
  );
}

// 結果の読み方の模式図: 同じ敵対的音声が、人間には「本人」、機械には「ターゲット話者」と受け取られる。位置は固定で、文字だけを変えられる。
// humanResult / machineResult は、2 行の文字の配列
export function Perception({ desc, source, human, machine, humanResult, machineResult }) {
  const { caption } = useContext(FigureContext);
  for (const [name, v] of Object.entries({ desc, source, human, machine })) if (!v) throw new Error(`<Perception> には ${name} が要ります`);
  if (humanResult?.length !== 2 || machineResult?.length !== 2) throw new Error("<Perception> の humanResult と machineResult は、2 行の文字の配列にします");
  for (const name of ["ear", "cpu"]) if (!icons.has(name)) throw new Error(`アイコン「${name}」が、site/assets/icons.svg にありません`);
  const t = (cls, x, y, text) => h("text", { class: cls, x, y, "text-anchor": "middle" }, text);
  const icon = (name, cls, x, y) => h("use", { class: `pc-icon ${cls}`, href: `assets/icons.svg#${name}`, x, y, width: 40, height: 40 });
  const bars = [10, 18, 30, 22, 34, 20, 28, 14, 8];
  return h(
    "svg",
    { class: "perception", viewBox: "0 0 400 340", role: "img", "aria-labelledby": "perception-title perception-desc" },
    h("title", { id: "perception-title" }, caption),
    h("desc", { id: "perception-desc" }, desc),
    h(
      "defs",
      null,
      h("marker", { id: "perception-arrow", viewBox: "0 0 10 10", refX: "8", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto" }, h("path", { class: "pc-arrowhead", d: "M0 0 L10 5 L0 10 z" })),
    ),
    // 分かれる前: 敵対的音声 (波形)
    h("rect", { class: "pc-source", x: 110, y: 8, width: 180, height: 84, rx: 14 }),
    bars.map((v, i) => h("line", { class: "pc-wave", key: i, x1: 140 + i * 15, y1: 36 - v / 2, x2: 140 + i * 15, y2: 36 + v / 2 })),
    t("pc-label", 200, 78, source),
    // 2 本に分かれる矢印
    h("path", { class: "pc-arrow", d: "M160 92 Q110 100 101 128", "marker-end": "url(#perception-arrow)" }),
    h("path", { class: "pc-arrow", d: "M240 92 Q290 100 299 128", "marker-end": "url(#perception-arrow)" }),
    // 人間 (左) と機械 (右)
    h("circle", { class: "pc-node-human", cx: 100, cy: 170, r: 36 }),
    icon("ear", "pc-icon-human", 80, 150),
    h("circle", { class: "pc-node-machine", cx: 300, cy: 170, r: 36 }),
    icon("cpu", "pc-icon-machine", 280, 150),
    t("pc-label", 100, 230, human),
    t("pc-label", 300, 230, machine),
    // 受け取り方
    h("path", { class: "pc-arrow", d: "M100 238 L100 258", "marker-end": "url(#perception-arrow)" }),
    h("path", { class: "pc-arrow", d: "M300 238 L300 258", "marker-end": "url(#perception-arrow)" }),
    h("rect", { class: "pc-result-human", x: 20, y: 262, width: 160, height: 64, rx: 12 }),
    h("rect", { class: "pc-result-machine", x: 220, y: 262, width: 160, height: 64, rx: 12 }),
    t("pc-label", 100, 288, humanResult[0]),
    t("pc-label", 100, 310, humanResult[1]),
    t("pc-label", 300, 288, machineResult[0]),
    t("pc-label", 300, 310, machineResult[1]),
  );
}

// ---- 文章の部品 -----------------------------------------------------------------------------

// 要点の 1 文 (左に線が付く段落)。live を付けると、中身が切り替わったときに、読み上げ機能へ知らせる
export function Summary({ live, children }) {
  return h("p", { class: "summary", ...(live ? { "aria-live": "polite" } : {}) }, children);
}

// 補足の 1 文 (小さい文字)。icon を付けると、文の頭にアイコンが付く
export function Hint({ icon, children }) {
  return h("p", { class: "hint" }, icon ? h(Icon, { name: icon, className: "icon icon-inline" }) : null, children);
}

// 注意の 1 文 (枠つき)
export function Notice({ icon = "info", children }) {
  return h("p", { class: "notice" }, h(Icon, { name: icon, className: "icon icon-inline" }), children);
}

// ---- 指標の説明、条件と限界 -------------------------------------------------------------------------

export function Legend({ children }) {
  return h("dl", { class: "legend" }, children);
}
export function Term({ name, children }) {
  if (!name) throw new Error("<Term> には name が要ります");
  return h("div", null, h("dt", null, name), h("dd", null, children));
}

export function Limits({ children }) {
  return h("ul", { class: "limits" }, children);
}
export function Limit({ title, children }) {
  if (!title) throw new Error("<Limit> には title が要ります");
  return h("li", { class: "limit" }, h("h4", null, title), h("p", null, children));
}

export const components = {
  Title, Lead, Toc, Block, Figure, Ref, Flow, Step, Branching, Alts, Case, MaskDemo, Perception,
  Summary, Hint, Notice, Legend, Term, Limits, Limit,
};
