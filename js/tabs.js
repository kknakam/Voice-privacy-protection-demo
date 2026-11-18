// ページ内のタブ (WAI-ARIA の tabs パターン)。
// - パネルは hidden で切り替えるだけなので、デモの録音・生成結果・入力は、別のタブを見ても失われない。
// - URL のハッシュで開ける (#demo / #scenario、またはパネル内の要素の id)。戻る・進むでもタブが切り替わる。
// - タブごとに、離れたときのスクロール位置を覚えておき、戻ったときに復元する。
// app.js とは独立に動かす (エンジンの読み込みが失敗しても、タブは使える)。

const list = document.querySelector('[role="tablist"]');
const tabs = list ? [...list.querySelectorAll('[role="tab"]')] : [];
const panels = new Map(tabs.map((tab) => [tab, document.getElementById(tab.getAttribute("aria-controls"))]));
const scrollMemory = new Map();
let current = tabs.find((tab) => tab.getAttribute("aria-selected") === "true") ?? tabs[0];

const nameOf = (tab) => panels.get(tab).id;

function tabContaining(element) {
  const panel = element?.closest('[role="tabpanel"]');
  return tabs.find((tab) => panels.get(tab) === panel) ?? null;
}

// 表示するタブを切り替える。切り替えたときだけ true を返す。
function show(tab) {
  if (tab === current) return false;
  scrollMemory.set(current, window.scrollY);
  panels.get(current).querySelectorAll("audio").forEach((audio) => audio.pause());
  for (const t of tabs) {
    const on = t === tab;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    panels.get(t).hidden = !on;
  }
  current = tab;
  document.dispatchEvent(new CustomEvent("tabchange", { detail: { tab: nameOf(tab) } }));
  return true;
}

function restoreScroll(tab) {
  if (scrollMemory.has(tab)) window.scrollTo({ top: scrollMemory.get(tab), behavior: "instant" });
  else panels.get(tab).scrollIntoView({ block: "start", behavior: "instant" });
}

// タブのボタンを押したとき、またはキーボードで選んだとき
function select(tab, { focus = false } = {}) {
  if (focus) tab.focus();
  if (!show(tab)) return;
  history.pushState(null, "", `#${nameOf(tab)}`);
  restoreScroll(tab);
}

// URL のハッシュに合わせる (読み込み時、戻る・進む、ページ内リンク)
function route() {
  const id = decodeURIComponent(location.hash.slice(1));
  const target = id ? document.getElementById(id) : null;
  // ハッシュが無いときは先頭のタブ。タブの外の要素 (#top など) を指すときは、何もしない
  const tab = id ? tabContaining(target) : tabs[0];
  if (!tab) return;
  if (!show(tab)) return;   // すでに表示中なら、スクロールはブラウザに任せる
  if (target && target !== panels.get(tab)) target.scrollIntoView({ block: "start", behavior: "instant" });
  else if (id) panels.get(tab).scrollIntoView({ block: "start", behavior: "instant" });
  else restoreScroll(tab);
}

function onKeydown(event) {
  const index = tabs.indexOf(document.activeElement);
  if (index < 0) return;
  const next = { ArrowRight: (index + 1) % tabs.length, ArrowLeft: (index - 1 + tabs.length) % tabs.length, Home: 0, End: tabs.length - 1 }[event.key];
  if (next === undefined) return;
  event.preventDefault();
  select(tabs[next], { focus: true });
}

if (tabs.length) {
  for (const tab of tabs) tab.addEventListener("click", () => select(tab));
  list.addEventListener("keydown", onKeydown);
  window.addEventListener("hashchange", route);
  route();
}
