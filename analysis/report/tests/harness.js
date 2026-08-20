// Minimal DOM stub: enough to run the report's drawing code and count output.
const TOKENS = {
  "--s1": "#2a78d6", "--s2": "#eb6834", "--s3": "#1baf7a", "--s4": "#7d5bbe",
  "--ink": "#14161a", "--ink-2": "#4e5560", "--ink-3": "#7b828d",
  "--grid": "#e6e6e1", "--rule": "#dcdcd6", "--surface-sunken": "#f2f2ef",
  "--slo": "#d64545",
};

let created = { svg: {}, html: {} };

function mkNode(tag, ns) {
  const n = {
    tag, ns, children: [], attrs: {}, _html: "", style: {}, listeners: {},
    setAttribute(k, v) { this.attrs[k] = v; },
    getAttribute(k) { return this.attrs[k]; },
    appendChild(c) { this.children.push(c); c.parent = this; return c; },
    removeChild(c) { this.children = this.children.filter((x) => x !== c); return c; },
    addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
    getBoundingClientRect() { return { left: 0, top: 0, width: 1000, height: 420 }; },
    get firstChild() { return this.children[0] || null; },
    set innerHTML(v) { this._html = v; this.children = []; },
    get innerHTML() { return this._html; },
    set textContent(v) { this._text = v; },
    get textContent() { return this._text; },
    get offsetWidth() { return 120; },
    get offsetHeight() { return 60; },
  };
  const bag = ns ? created.svg : created.html;
  bag[tag] = (bag[tag] || 0) + 1;
  return n;
}

const byId = {};
for (const id of ["arm-toggles", "cdf-tpot", "cdf-ttft", "cdf-len", "ts-stack", "ts-legend", "trace-stats", "trace-chart", "pareto", "batch-steps", "steptype-legend"]) {
  byId[id] = mkNode("div");
}
function sel(id, value) {
  const n = mkNode("select");
  n.value = value;
  byId[id] = n;
  return n;
}
sel("tpot-scale", "lin"); sel("tpot-clip", "1");
sel("ttft-scale", "lin"); sel("ttft-clip", "1");
sel("ts-scale", "shared"); sel("ts-smooth", "1");
sel("len-metric", "isl");
sel("tr-bin", "1");
sel("pf-thru", "tot"); sel("pf-agg", "mean");

global.document = {
  documentElement: mkNode("html"),
  getElementById: (id) => byId[id] || null,
  createElement: (t) => mkNode(t),
  createElementNS: (ns, t) => mkNode(t, ns),
};
global.getComputedStyle = () => ({ getPropertyValue: (k) => TOKENS[k] || "#000" });
global.window = { matchMedia: () => ({ addEventListener() {} }) };
global.MutationObserver = class { observe() {} };

require("./script.js");

// ---- assertions ----
function countDeep(node, pred, acc = { n: 0 }) {
  if (pred(node)) acc.n++;
  for (const c of node.children) countDeep(c, pred, acc);
  return acc.n;
}
const isPath = (n) => n.tag === "path" && n.attrs.d && n.attrs.stroke && n.attrs.stroke !== "none";
const isSvg = (n) => n.tag === "svg";

let fail = 0;
function check(name, got, want) {
  const ok = got === want;
  if (!ok) fail++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}: got ${got}${ok ? "" : ", want " + want}`);
}
function checkMin(name, got, min) {
  const ok = got >= min;
  if (!ok) fail++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}: got ${got}${ok ? "" : ", want >= " + min}`);
}

check("arm toggle buttons", byId["arm-toggles"].children.length, 4);
check("pareto sub-charts", countDeep(byId["pareto"], isSvg), 2);
check("pareto markers (4 arms x 2 charts)", countDeep(byId["pareto"], (n) => n.tag === "circle"), 8);
check("trace stat chips", byId["trace-stats"].children.length, 7);
check("trace panel svgs", countDeep(byId["trace-chart"], isSvg), 2);
checkMin("trace stroked paths", countDeep(byId["trace-chart"], isPath), 2);
console.log("   chips:", byId["trace-stats"].children.map(function(li){return li.innerHTML.replace(/<[^>]+>/g," ").trim();}).join(" | "));
for (const id of ["cdf-tpot", "cdf-ttft", "cdf-len"]) {
  check(`${id} svg count`, countDeep(byId[id], isSvg), 1);
  checkMin(`${id} stroked paths (>=4 curves)`, countDeep(byId[id], isPath), 4);
}
check("stack panel svgs", countDeep(byId["ts-stack"], isSvg), 7);
// 6 data panels x 4 arms = 24 lines, + running band fill, + arrivals line/area
checkMin("stack stroked paths", countDeep(byId["ts-stack"], isPath), 24);
check("ts legend entries", byId["ts-legend"].children.length, 5);

// pointermove must not throw and must populate a tooltip
const tpot = byId["cdf-tpot"];
const cdfSvg = tpot.children.find(isSvg);
cdfSvg.listeners.pointermove.forEach((f) => f({ clientX: 500, clientY: 200 }));
const tip = tpot.children.find((c) => c.tag === "div");
checkMin("cdf tooltip populated", tip.innerHTML.length, 20);
console.log("   tooltip:", tip.innerHTML.replace(/<[^>]+>/g, "").trim().slice(0, 90));

// panel boxes only -- the last child div is the tooltip, which has no svg
const stackSvgs = byId["ts-stack"].children
  .filter((c) => c.tag === "div" && c.children[0] && c.children[0].tag === "svg")
  .map((d) => d.children[0]);
stackSvgs.forEach((sv) => sv.listeners.pointermove.forEach((f) => f({ clientX: 500, clientY: 60 })));
const stip = byId["ts-stack"].children.find((c) => c.tag === "div" && c.listeners.pointermove === undefined && c._html);
checkMin("stack tooltip populated", stip ? stip.innerHTML.length : 0, 20);
console.log("   tooltip:", (stip ? stip.innerHTML : "").replace(/<[^>]+>/g, " ").trim().slice(0, 110));


// ---- value labels at the crosshair ----
// Every panel must report a number, not just the hovered one.
const isText = (n) => n.tag === "text";
function marksOf(svg) {           // the <g> the crosshair markers live in
  const gs = svg.children.filter((c) => c.tag === "g");
  return gs[gs.length - 1];
}
const cdfMarks = marksOf(cdfSvg);
check("cdf value labels (one per arm)", countDeep(cdfMarks, isText), 4);
console.log("   labels:", cdfMarks.children.filter(isText).map((t) => t.textContent).join(" "));

let labelled = 0, panelsWithDots = 0;
for (const sv of stackSvgs) {
  const m = marksOf(sv);
  const dots = countDeep(m, (n) => n.tag === "circle");
  const texts = countDeep(m, isText);
  if (dots) panelsWithDots++;
  if (dots && texts === dots) labelled++;
}
check("stack panels showing dots", panelsWithDots, 7);
check("stack panels labelling every dot", labelled, 7);
const runMarks = marksOf(stackSvgs[4]);
console.log("   running-requests labels:",
  runMarks.children.filter(isText).map((t) => t.textContent).join(" "));

// ---- 5.1 engine step-type bars (table 1 only; table 2 stays a table) ----
check("step-type svg", countDeep(byId["batch-steps"], isSvg), 1);
check("step-type bars (4 arms x 3 types)",
  countDeep(byId["batch-steps"], (n) => n.tag === "rect"), 12);
check("step-type legend entries", byId["steptype-legend"].children.length, 3);
const stSvg = byId["batch-steps"].children.find(isSvg);
stSvg.listeners.pointermove.forEach((f) => f({ clientX: 150, clientY: 300 }));
const stTip = byId["batch-steps"].children.find((c) => c.tag === "div");
checkMin("step-type tooltip populated", stTip.innerHTML.length, 20);
console.log("   tooltip:", stTip.innerHTML.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim().slice(0, 80));

console.log(fail ? `\n${fail} CHECK(S) FAILED` : "\nALL CHECKS PASSED");
process.exit(fail ? 1 : 0);
