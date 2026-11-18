// 日本語の文章 -> OpenJTalk (WASM) -> アクセント結合の前の独自ルール -> フルコンテキストラベル -> 音素 + 韻律記号 の ID 列
//
// 学習時の音素化は pyopenjtalk-plus を使っていた。ブラウザでは、同じ OpenJTalk の C のコアを WASM にしたものを使い、
// pyopenjtalk-plus が Python で行っていた「アクセント結合の前の独自ルール」のうち、数詞の読み替えを除く部分を、ここに移した。
// Sudachi による読みの補正などの後処理は含まない (そのため、学習時と完全には一致しない文がある)。

import { labelsToSymbols, symbolsToIds } from "./ja_g2p.js";

const DIGITS = "０１２３４５６７８９〇零一二三四五六七八九";
const DASHES = "−－ー‐‑‒–—-";
const HIRAGANA = /[ぁ-ゖ]/;
const HAN = /[一-鿿]/;

const FIELDS = ["string", "pos", "pos_group1", "pos_group2", "pos_group3", "ctype", "cform", "orig", "read", "pron", "acc", "mora_size", "chain_rule", "chain_flag"];
const INT_FIELDS = new Set(["acc", "mora_size", "chain_flag"]);

function parseTsv(tsv) {
  return tsv.split("\n").filter(Boolean).map((line) => {
    const cols = line.split("\t");
    const f = {};
    FIELDS.forEach((name, k) => (f[name] = INT_FIELDS.has(name) ? parseInt(cols[k], 10) : cols[k] ?? ""));
    return f;
  });
}

function toTsv(features) {
  return features.map((f) => FIELDS.map((name) => String(f[name])).join("\t")).join("\n") + "\n";
}

// pyopenjtalk-plus の _apply_original_rule_before_chaining (modify_numeral_reading=False の場合) と同じ規則
export function applyOriginalRule(f) {
  const zeroIdx = new Set();
  let index = 0;
  while (index < f.length) {
    const start = index;
    while (index < f.length && f[index].string.length === 1 && DIGITS.includes(f[index].string)) index++;
    if (index === start) {
      index++;
      continue;
    }
    // 「一〇〇一号室」の〇は番号の桁なので、伏字の「〇〇町」と区別して数詞へ渡す
    let isNumber = f.slice(start, index).some((x) => x.string !== "〇");
    if (index < f.length) {
      const following = f[index];
      isNumber = isNumber || following.pos_group2 === "助数詞";
      if (DASHES.includes(following.string) && index + 1 < f.length) isNumber = isNumber || DIGITS.includes(f[index + 1].string);
    }
    if (start >= 2 && (DASHES + "，").includes(f[start - 1].string)) isNumber = isNumber || DIGITS.includes(f[start - 2].string);
    if (isNumber && index - start > 1) {
      for (let k = start; k < index; k++) {
        const z = f[k];
        if (z.string === "〇") {
          zeroIdx.add(k);
          z.pos = "名詞";
          z.pos_group1 = "数";
          z.read = z.pron = "ゼロ";
          z.mora_size = 2;
        }
      }
    }
  }

  for (let i = 0; i < f.length - 1; i++) {
    const njd = f[i];
    const next = f[i + 1];
    // ナイ形に続く「の」「と」は、助動詞に付く結合規則を明示して否定形の核を保つ
    if (
      njd.pos === "助動詞" && njd.orig === "ない" &&
      (next.pos === "助詞" || (next.pos === "名詞" && next.pos_group1 === "非自立")) &&
      (next.orig === "の" || next.orig === "と")
    ) {
      next.chain_rule = "助動詞%F1/" + next.chain_rule.split("/").filter((r) => !r.startsWith("助動詞%")).join("/");
    }
    // 「放っておく」「放っとく」の「放っ」は平板型
    if (
      njd.string === "放っ" &&
      ((next.string === "て" && i + 2 < f.length && f[i + 2].orig === "おく" && f[i + 2].pos === "動詞") ||
        (next.orig === "とく" && next.pos === "動詞"))
    ) {
      njd.acc = 0;
    }
    // 動詞の意向形の「う」は、付属語の「う」が核を決める型にする
    if (njd.pos === "動詞" && njd.cform === "未然ウ接続" && next.pos === "助動詞" && next.orig === "う") {
      const flatBeforeToSuru = njd.acc === 0 && i + 3 < f.length && f[i + 2].string === "と" && (f[i + 3].orig === "する" || f[i + 3].orig === "思う");
      if (!flatBeforeToSuru) next.chain_rule = "動詞%F4@0/特殊助動詞%F2@0";
    }
    // 平板の動詞に使役の「せ」「させ」と「ない」が続く場合は平板のまま
    if (
      njd.pos === "動詞" && njd.pos_group1 === "自立" && njd.acc === 0 &&
      next.pos === "動詞" && next.pos_group1 === "接尾" && (next.orig === "せる" || next.orig === "させる") &&
      i + 2 < f.length && f[i + 2].pos === "助動詞" && f[i + 2].orig === "ない"
    ) {
      next.acc = 0;
      f[i + 2].chain_rule = f[i + 2].chain_rule.split("/").map((r) => (r.startsWith("動詞%") ? "動詞%F1" : r)).join("/");
    }
    if (
      njd.pos === "動詞" && njd.pos_group1 === "自立" && njd.acc === 0 &&
      next.pos === "動詞" && next.pos_group1 === "接尾" && (next.orig === "せる" || next.orig === "させる") &&
      next.cform === "基本形"
    ) {
      next.acc = 0;
    }
    // 4 モーラ以上の名詞に続く「湖」は「ズウミ」ではなく別のアクセント句
    if (njd.pos === "名詞" && next.string === "湖" && next.pron === "ミズウミ") {
      let moras = 0;
      for (let p = i; p >= 0; p--) {
        const prev = f[p];
        if (prev.pos !== "名詞") break;
        moras += prev.mora_size;
        if (prev.chain_flag === 0) break;
      }
      if (moras + 1 >= 4) next.chain_rule = "C3";
    }
    // 名詞の後ろで新しい語を作る「不足」は連濁した「ブソク」
    if (njd.pos === "名詞" && next.string === "不足" && next.pron === "フソク") {
      next.read = "ブソク";
      next.pron = "ブソク";
    }
    // 「〜玉」の「球」
    if (
      next.string === "球" && next.pos === "名詞" && next.pos_group1 === "接尾" && next.pron === "キュー" &&
      [...njd.string].some((c) => HAN.test(c)) && HIRAGANA.test(njd.string)
    ) {
      next.read = "ダマ";
      next.pron = "ダマ";
      next.acc = 1;
      next.mora_size = 2;
      next.chain_rule = "C4";
    }
    // サ変動詞 (スル) の前にサ変接続や名詞が来た場合は、一つのアクセント句に纏める
    if (
      (["サ変接続", "格助詞", "接続助詞"].includes(njd.pos_group1) || (njd.pos === "名詞" && njd.pos_group1 === "一般") || njd.pos === "副詞") &&
      next.ctype === "サ変・スル"
    ) {
      next.chain_flag = 1;
    }
    // 接頭辞「お」「御」「ご」
    if (["お", "御", "ご"].includes(njd.string) && njd.chain_rule === "P1") {
      if (next.acc === 0 || next.acc === next.mora_size) {
        next.chain_rule = "C4";
        next.acc = 0;
      } else {
        next.chain_rule = "C1";
      }
    }
    // 動詞の連続
    if (njd.pos === "動詞" && next.pos === "動詞") next.chain_rule = next.acc !== 0 ? "C1" : "C4";
    // 連用形の語尾の核
    if (["連用形", "連用タ接続", "連用ゴザイ接続", "連用テ接続"].includes(njd.cform) && njd.acc === njd.mora_size && njd.mora_size > 1) njd.acc -= 1;
    // 受身・使役・「ちゃう」に続く「た」
    if (["れる", "られる", "せる", "させる", "ちゃう"].includes(njd.orig) && next.string === "た") next.chain_rule = "F2@1";
    // 形容詞に続く「なる」「する」
    if (njd.pos === "形容詞" && (next.orig === "なる" || next.orig === "する")) next.chain_flag = 1;
  }

  // フィラーは単独のアクセント句にする。直後の名詞とも分ける
  for (let i = 0; i < f.length; i++) {
    if (f[i].pos !== "フィラー") continue;
    f[i].chain_flag = 0;
    if (i + 1 < f.length && f[i + 1].pos === "名詞") f[i + 1].chain_flag = 0;
  }
  return f;
}

export class JaFrontend {
  // module: createOpenJTalk() の結果。辞書は FS の dicDir に置いてあること
  constructor(module, dicDir = "/dic") {
    this.m = module;
    const put = (s) => {
      const n = module.lengthBytesUTF8(s) + 1;
      const p = module._malloc(n);
      module.stringToUTF8(s, p, n);
      return p;
    };
    this.put = put;
    const dp = put(dicDir);
    this.ready = module._jt_init(dp) === 1;
    module._free(dp);
  }

  call(fn, text) {
    const p = this.put(text);
    const r = fn(p);
    this.m._free(p);
    return r ? this.m.UTF8ToString(r) : null;
  }

  // 規則なし (OpenJTalk 本体と同じ流れ)
  labelsPlain(text) {
    const out = this.call(this.m._jt_fullcontext, text);
    return out === null ? null : out.split("\n").filter(Boolean);
  }

  // 独自ルールあり
  labels(text) {
    const tsv = this.call(this.m._jt_stage1, text);
    if (tsv === null) return null;
    const features = parseTsv(tsv);
    if (features.length === 0) return null;
    const out = this.call(this.m._jt_stage2, toTsv(applyOriginalRule(features)));
    return out === null ? null : out.split("\n").filter(Boolean);
  }

  // 文章 -> { ids, symbols } または { error }
  encode(text, ja) {
    const labels = this.labels(text);
    if (!labels || labels.length < 3) return { error: "この文章は読み上げに変換できませんでした。" };
    const symbols = labelsToSymbols(labels);
    const ids = symbolsToIds(symbols, ja);
    if (!ids) return { error: "モデルが扱えない記号が含まれています。" };
    return { ids, symbols };
  }
}
