// 「研究のシナリオ」の MDX (content/scenario.mdx) で使うコンポーネント。
// HTML は、site/css/style.css のクラスだけで書く。CSP (style-src 'self') のため、style 属性は使わない。
// ビルド時 (Node) に、HTML の文字列へ変換するだけで、ブラウザには配信しない。
import { createContext, h, toChildArray } from "preact";
import { useContext } from "preact/hooks";

const COLORS = new Set(["purple", "teal", "orange", "gray"]);   // 紫 = 本人、青緑 = 保護、橙 = 攻撃者、灰 = 中立

let icons = new Set();
let figureNo = 0;

// ビルドのたびに呼ぶ。icons は、site/assets/icons.svg にあるアイコンの id の集合
export function setup(options) {
  icons = options.icons;
  figureNo = 0;
}

const FigureContext = createContext({ caption: "" });

// 空白だけの文字列 (JSX の行の間の改行) を除いた、子要素の配列
const kids = (children) => toChildArray(children).filter((c) => !(typeof c === "string" && c.trim() === ""));

function Icon({ name, hidden = true }) {
  if (!icons.has(name)) throw new Error(`アイコン「${name}」が、site/assets/icons.svg にありません`);
  return h("svg", { class: "icon", ...(hidden ? { "aria-hidden": "true" } : {}) }, h("use", { href: `assets/icons.svg#${name}` }));
}

// 節。id は、目次やリンクの行き先になる。見出しの id は「<id>-title」
export function Block({ id, eyebrow, title, children }) {
  if (!id || !eyebrow || !title) throw new Error("<Block> には、id・eyebrow・title が要ります");
  return h(
    "section",
    { class: "block", id, "aria-labelledby": `${id}-title` },
    h("div", null, h("span", { class: "eyebrow" }, eyebrow), h("h3", { id: `${id}-title` }, title)),
    children,
  );
}

// 図。番号は、本文に出てくる順に自動で振る
export function Figure({ caption, children }) {
  if (!caption) throw new Error("<Figure> には caption が要ります");
  figureNo += 1;
  return h(
    "figure",
    { class: "diagram" },
    h(FigureContext.Provider, { value: { caption } }, children),
    h("figcaption", null, `図 ${figureNo}　${caption}`),
  );
}

// 流れの図。<Step> を並べると、間の矢印は自動で入る。label を省くと、図の caption が、読み上げ用の名前になる
export function Flow({ label, children }) {
  const { caption } = useContext(FigureContext);
  const steps = kids(children);
  if (steps.length === 0) throw new Error("<Flow> には、<Step> が 1 つ以上要ります");
  return h(
    "ol",
    { class: "flow", "aria-label": label ?? caption },
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

// 流れの 1 手順。role は、役割の文字 (色だけに頼らないため)。本文は 1 行の文
export function Step({ role, icon, color, title, children }) {
  if (!COLORS.has(color)) throw new Error(`<Step> の color「${color}」は使えません (${[...COLORS].join(" / ")})`);
  if (!role || !icon || !title) throw new Error(`<Step title="${title}"> には、role・icon・title が要ります`);
  return h(
    "div",
    { class: "flow-card" },
    h("span", { class: `sticker sticker-${color}` }, h(Icon, { name: icon })),
    h("div", { class: "flow-body" }, h("span", { class: "flow-role" }, role), h("h4", null, title), h("p", null, children)),
  );
}

// 要点の 1 文 (左に線が付く段落)
export function Summary({ children }) {
  return h("p", { class: "summary" }, children);
}

export const components = { Block, Figure, Flow, Step, Summary };
