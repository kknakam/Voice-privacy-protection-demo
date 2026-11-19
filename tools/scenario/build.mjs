// 本文 (MDX) を HTML にして、site/index.html の印の間に書き込む。
//   content/scenario.mdx  → 「研究のシナリオ」のタブ (印: scenario-content)
//   content/demo.mdx      → 「デモ」のタブ         (印: demo-content)
//
//   node tools/scenario/build.mjs                       どちらも書き込む
//   node tools/scenario/build.mjs --check               書き込まずに、index.html が MDX と一致しているかを調べる (食い違うと、終了コード 1)
//   --only scenario|demo                                片方だけ
//   --source <ファイル>                                  --only と合わせて、MDX の代わりに、別の MDX を使う (試すとき)
//
// 印: <!-- BEGIN <名前> ... --> と <!-- END <名前> --> の間は、自動生成。手で直さない。
// 書き込んだあと、site/js の JS が探す id が、HTML に全部あるかも調べる (デモのタブの部品は、id で JS とつながるため)。
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { evaluate } from "@mdx-js/mdx";
import { h } from "preact";
import * as runtime from "preact/jsx-runtime";
import { renderToString } from "preact-render-to-string";
import { collected, components, setup } from "./components.mjs";
import { demoComponents } from "./demo-components.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const args = process.argv.slice(2);
const flagValue = (name) => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined);
const TARGET = path.join(root, "site/index.html");
const SPRITE = path.join(root, "site/assets/icons.svg");
const INDENT = 8;   // index.html で、節や、ステップが並ぶ位置の字下げ

const DOCS = {
  scenario: { source: "content/scenario.mdx", marker: "scenario-content", components },
  demo: { source: "content/demo.mdx", marker: "demo-content", components: { ...components, ...demoComponents } },
};

// 枠になるタグだけを、行に分けて字下げする。段落や見出しの中の空白は、触らない
const CONTAINERS = new Set(["section", "article", "header", "div", "figure", "ol", "ul", "li", "nav", "dl", "fieldset"]);
function format(html) {
  const out = [];
  let depth = 0;
  let line = "";
  const pad = () => " ".repeat(INDENT + depth * 2);
  const flush = () => {
    if (line) out.push(pad() + line);
    line = "";
  };
  for (const token of html.match(/<!--[\s\S]*?-->|<[^>]+>|[^<]+/g) ?? []) {
    const tag = token.match(/^<(\/)?([A-Za-z][A-Za-z0-9]*)/);
    if (tag && CONTAINERS.has(tag[2])) {
      flush();
      if (tag[1]) depth -= 1;
      out.push(pad() + token);
      if (!tag[1] && !token.endsWith("/>")) depth += 1;
    } else if (line || token.trim()) {
      line += token;   // 枠のタグの間にある、空白だけの文字列 (MDX が入れる改行) は捨てる
    }
  }
  flush();
  return out.join("\n") + "\n";
}

// MDX を、HTML の文字列にする。known は、前の回に集めた、図の番号と節の一覧 (1 回目は null)
async function renderOnce(source, doc, icons, known) {
  setup({ icons, known, siteRoot: path.join(root, "site") });
  const { default: Content } = await evaluate(source, { ...runtime, development: false });
  return renderToString(h(Content, { components: doc.components }));
}

async function render(doc, sourcePath) {
  const sprite = fs.readFileSync(SPRITE, "utf8");
  const icons = new Set([...sprite.matchAll(/<symbol id="([^"]+)"/g)].map((m) => m[1]));
  const source = fs.readFileSync(sourcePath, "utf8");
  // 目次や「図 N」の参照は、あとに出てくる節や図を指せるので、2 回描画する。1 回目で集め、2 回目で使う
  await renderOnce(source, doc, icons, null);
  const html = await renderOnce(source, doc, icons, collected());
  // CSP (style-src 'self'、script-src 'self') で、実行や適用が止められるものは、生成の段階で止める
  if (/\sstyle=/.test(html)) throw new Error("style 属性は使えません (CSP でブロックされます)。色や余白は、CSS のクラスで指定してください");
  if (/<script[\s>]/i.test(html) || /\son[a-z]+=/i.test(html)) throw new Error("<script> やイベント属性 (onclick など) は使えません (CSP でブロックされます)");
  return format(html);
}

// JS が id で探す部品が、HTML に全部あるか。JS の文字列から、id らしいもの (接頭辞つき) を拾って、突き合わせる
const ID_PREFIX = "(?:record|attack|clone|compare|heat|step|steps|env|tab|scenario|mask)";
function checkIds(html) {
  const ids = new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]));
  const problems = [];
  for (const file of ["site/js/demo/app.js", "site/js/site/tabs.js", "site/js/site/scenario.js"]) {
    const text = fs.readFileSync(path.join(root, file), "utf8").replace(/\/\/.*$/gm, "");
    const exact = new Set([...text.matchAll(new RegExp(`(["'])(${ID_PREFIX}-[a-z0-9-]+)\\1`, "g"))].map((m) => m[2]));
    for (const id of exact) if (!ids.has(id) && !IGNORED.has(id)) problems.push(`${file} が探す id「${id}」が、HTML にありません`);
    for (const m of text.matchAll(new RegExp(`\`(${ID_PREFIX}-[a-z0-9-]*)\\$\\{`, "g"))) {
      if (![...ids].some((id) => id.startsWith(m[1]))) problems.push(`${file} が探す id「${m[1]}…」に当たる id が、HTML にありません`);
    }
  }
  return problems;
}
// id ではなく、クラス名などの、同じ形の文字列
const IGNORED = new Set(["step-head", "tab-busy", "mask-keep", "mask-cut"]);

const checkOnly = args.includes("--check");
const only = flagValue("--only");
if (only && !DOCS[only]) {
  console.error(`--only には、${Object.keys(DOCS).join(" か ")} を指定してください`);
  process.exit(2);
}
const sourceArg = flagValue("--source");
if (sourceArg && !only) {
  console.error("--source は、--only と合わせて指定してください");
  process.exit(2);
}

let html = fs.readFileSync(TARGET, "utf8");
const original = html;
const summary = [];
for (const [name, doc] of Object.entries(DOCS)) {
  if (only && only !== name) continue;
  const begin = new RegExp(`^[ \\t]*<!-- BEGIN ${doc.marker}[^\\n]*-->[ \\t]*\\n`, "m").exec(html);
  const end = new RegExp(`^[ \\t]*<!-- END ${doc.marker} -->`, "m").exec(html);
  if (!begin || !end || end.index < begin.index) {
    console.error(`site/index.html に、<!-- BEGIN ${doc.marker} ... --> と <!-- END ${doc.marker} --> の印がありません`);
    process.exit(2);
  }
  const sourcePath = sourceArg ? path.resolve(sourceArg) : path.join(root, doc.source);
  let generated;
  try {
    generated = await render(doc, sourcePath);
  } catch (e) {
    // MDX の書き間違いは、行番号つきで、1 行にまとめて知らせる
    const reason = e.reason ?? e.message;
    const where = e.line && !/\(\d+:\d+/.test(reason) ? ` (${path.basename(sourcePath)} の ${e.line} 行目)` : "";
    console.error(`エラー: ${reason}${where}`);
    process.exit(1);
  }
  html = html.slice(0, begin.index + begin[0].length) + generated + html.slice(end.index);
  summary.push(`${name}: 生成 ${generated.split("\n").length - 1} 行`);
}

if (!only) {
  const problems = checkIds(html);
  if (problems.length) {
    console.error(["エラー: JS が探す id と、HTML が合っていません:", ...problems.map((p) => `  ${p}`)].join("\n"));
    process.exit(1);
  }
}

if (checkOnly) {
  if (html === original) {
    console.log("一致: site/index.html は、content/*.mdx から生成した内容と同じです");
  } else {
    console.error("不一致: site/index.html の印の間が、content/*.mdx と食い違っています。npm run build を実行してください");
    process.exit(1);
  }
} else {
  fs.writeFileSync(TARGET, html);
  console.log(`更新: site/index.html (${summary.join("、")}、図 ${(html.match(/<figure class="diagram"/g) ?? []).length} 枚)`);
}
