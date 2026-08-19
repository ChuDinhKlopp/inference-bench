// Extract PARETO + paretoFront from the built script and confirm the report's
// dominance claim for every throughput x latency x statistic combination.
const src = require('fs').readFileSync('script.js', 'utf8');
const grab = (name, open, close) => {
  const i = src.indexOf(name);
  const s = src.indexOf(open, i);
  let d = 0, j = s;
  for (; j < src.length; j++) {
    if (src[j] === open) d++;
    else if (src[j] === close) { d--; if (!d) break; }
  }
  return src.slice(s, j + 1);
};
const PARETO = eval('(' + grab('const PARETO =', '{', '}') + ')');
const paretoFront = eval('(function paretoFront' + grab('function paretoFront', '(', ')') + grab('function paretoFront(pts) {', '{', '}') + ')');

const arms = Object.keys(PARETO);
// The report claims: sole-optimal w16kv16 in 21 of 24 combinations; the only
// exceptions are the three p50-TTFT pairings, where w8kv16 also makes the front.
let sole = 0, exceptions = [];
for (const thru of ['tot', 'out', 'req']) {
  for (const lat of ['ttft', 'tpot']) {
    for (const agg of ['mean', 'p50', 'p90', 'p99']) {
      const pts = arms.map((k) => ({ k, x: PARETO[k][lat][agg], y: PARETO[k][thru] }));
      const front = paretoFront(pts);
      const opt = pts.filter((_, i) => front[i]).map((p) => p.k);
      if (opt.length === 1 && opt[0] === 'w16kv16') sole++;
      else exceptions.push({ combo: `${thru} x ${agg} ${lat}`, opt });
    }
  }
}
let fail = 0;
const eq = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
const want = ['w16kv16', 'w8kv16'].sort();
function assert(name, ok, detail) {
  if (!ok) fail++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ': ' + detail : ''}`);
}
assert('sole-optimal count is 21 of 24', sole === 21, `got ${sole}`);
assert('exactly 3 exceptions', exceptions.length === 3, `got ${exceptions.length}`);
assert('every exception is a p50 TTFT pairing',
  exceptions.every((e) => e.combo.includes('p50 ttft')));
assert('every exception frontier is {w16kv16, w8kv16}',
  exceptions.every((e) => eq(e.opt.slice().sort(), want)));
// the margin the report quotes
const dLat = (PARETO.w16kv16.ttft.p50 - PARETO.w8kv16.ttft.p50) / PARETO.w16kv16.ttft.p50 * 100;
const dThr = (PARETO.w16kv16.tot - PARETO.w8kv16.tot) / PARETO.w16kv16.tot * 100;
assert('quoted margin: 0.26% latency gain', dLat.toFixed(2) === '0.26', dLat.toFixed(3) + '%');
assert('quoted margin: 2.6% throughput loss', dThr.toFixed(1) === '2.6', dThr.toFixed(3) + '%');
console.log(fail ? `\n${fail} FAILED` : '\nPARETO CLAIMS MATCH THE DATA');
process.exit(fail ? 1 : 0);
