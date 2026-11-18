// 「研究のシナリオ」を、content/scenario.mdx から HTML にして、site/index.html の印の間に書き込む。
//
//   node tools/scenario/build.mjs           書き込む
//   node tools/scenario/build.mjs --check   書き込まずに、index.html が MDX と一致しているかを調べる (食い違うと、終了コード 1)
//   --source <ファイル>                      content/scenario.mdx の代わりに、別の MDX を使う (--check と合わせて、試すときに使う)
//
// 印: <!-- BEGIN scenario-content ... --> と <!-- END scenario-content --> の間は、自動生成。手で直さない。
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { evaluate } from "@mdx-js/mdx";
import { h } from "preact";
import * as runtime from "preact/jsx-runtime";
import { renderToString } from "preact-render-to-string";
import { components, setup } from "./components.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const sourceArg = process.argv.indexOf("--source");
const SOURCE = sourceArg > 0 ? path.resolve(process.argv[sourceArg + 1]) : path.join(root, "content/scenario.mdx");
const TARGET = path.join(root, "site/index.html");
const SPRITE = path.join(root, "site/assets/icons.svg");
const BEGIN = /^[ \t]*<!-- BEGIN scenario-content[^\n]*-->[ \t]*\n/m;
const END = /^[ \t]*<!-- END scenario-content -->/m;
const INDENT = 8;   // index.html で、節 (section.block) が並ぶ位置の字下げ

// 枠になるタグだけを、行に分けて字下げする。段落や見出しの中の空白は、触らない
const CONTAINERS = new Set(["section", "div", "figure", "ol", "ul", "li", "nav", "dl"]);
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

async function render() {
  const sprite = fs.readFileSync(SPRITE, "utf8");
  setup({ icons: new Set([...sprite.matchAll(/<symbol id="([^"]+)"/g)].map((m) => m[1])) });
  const { default: Content } = await evaluate(fs.readFileSync(SOURCE, "utf8"), { ...runtime, development: false });
  const html = renderToString(h(Content, { components }));
  // CSP (style-src 'self'、script-src 'self') で、実行や適用が止められるものは、生成の段階で止める
  if (/\sstyle=/.test(html)) throw new Error("style 属性は使えません (CSP でブロックされます)。色や余白は、CSS のクラスで指定してください");
  if (/<script[\s>]/i.test(html) || /\son[a-z]+=/i.test(html)) throw new Error("<script> やイベント属性 (onclick など) は使えません (CSP でブロックされます)");
  return format(html);
}

const checkOnly = process.argv.includes("--check");
const html = fs.readFileSync(TARGET, "utf8");
const begin = BEGIN.exec(html);
const end = END.exec(html);
if (!begin || !end || end.index < begin.index) {
  console.error("site/index.html に、<!-- BEGIN scenario-content ... --> と <!-- END scenario-content --> の印がありません");
  process.exit(2);
}
let generated;
try {
  generated = await render();
} catch (e) {
  // MDX の書き間違いは、行番号つきで、1 行にまとめて知らせる
  const reason = e.reason ?? e.message;
  const where = e.line && !/\(\d+:\d+/.test(reason) ? ` (${path.basename(SOURCE)} の ${e.line} 行目)` : "";
  console.error(`エラー: ${reason}${where}`);
  process.exit(1);
}
const next = html.slice(0, begin.index + begin[0].length) + generated + html.slice(end.index);

if (checkOnly) {
  if (next === html) {
    console.log("一致: site/index.html は、content/scenario.mdx から生成した内容と同じです");
  } else {
    console.error("不一致: site/index.html の印の間が、content/scenario.mdx と食い違っています。npm run build:scenario を実行してください");
    process.exit(1);
  }
} else {
  fs.writeFileSync(TARGET, next);
  console.log(`更新: site/index.html (生成 ${generated.split("\n").length - 1} 行、図 ${(generated.match(/<figure /g) ?? []).length} 枚)`);
}
