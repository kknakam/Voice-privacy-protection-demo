// 日本語の折り返し (Safari など、CSS の word-break: auto-phrase が使えないブラウザ向けの代わり)。
// 文を文節に分けて (BudouX、Apache-2.0。site/vendor/budoux/)、文節の間に <wbr> を入れる。
// 文節の途中では折り返さないよう、<html> に budoux のクラスを付けて、CSS の word-break: keep-all を有効にする。
// auto-phrase が使えるブラウザ (Chrome、Edge) では、何もしない。
const supported = window.CSS?.supports?.("word-break", "auto-phrase");

const JAPANESE = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]/;
// 入力欄や、図の中の文字 (SVG は 1 行だけ) には、手を入れない
const SKIP = "script, style, textarea, input, select, option, svg, [data-keep-text]";

async function start() {
  const [{ Parser }, { model }] = await Promise.all([import("../../vendor/budoux/parser.js"), import("../../vendor/budoux/ja.js")]);
  const parser = new Parser(model);
  const split = (n) => n.previousSibling?.className === "bx" || n.nextSibling?.className === "bx";   // 分けたあとの文字には、もう一度手を入れない

  const process = (root) => {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => (JAPANESE.test(n.nodeValue) && !split(n) && !n.parentElement?.closest(SKIP) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT),
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      const chunks = parser.parse(node.nodeValue);
      if (chunks.length < 2) continue;
      const frag = document.createDocumentFragment();
      chunks.forEach((chunk, i) => {
        if (i > 0) frag.appendChild(Object.assign(document.createElement("wbr"), { className: "bx" }));
        frag.appendChild(document.createTextNode(chunk));
      });
      node.replaceWith(frag);
    }
  };

  const observer = new MutationObserver((records) => {
    const targets = new Set();
    for (const r of records) {
      if (r.type === "characterData") targets.add(r.target.parentElement);
      for (const added of r.addedNodes) {
        if (added.nodeType === Node.ELEMENT_NODE) targets.add(added);
        else if (added.nodeType === Node.TEXT_NODE && !split(added)) targets.add(added.parentElement);
      }
    }
    observer.disconnect();                      // 自分の書き換えには反応しない
    for (const el of targets) if (el?.isConnected) process(el);
    observer.takeRecords();
    observer.observe(document.body, { childList: true, subtree: true, characterData: true });
  });

  process(document.body);
  document.documentElement.classList.add("budoux");
  observer.observe(document.body, { childList: true, subtree: true, characterData: true });
}

if (!supported) start().catch((e) => console.warn("日本語の折り返しの補助を読み込めませんでした:", e));
