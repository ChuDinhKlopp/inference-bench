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
// The report claims (MI250 run, 2026-08-18): w16kv16 is sole-optimal in all
// 24 of 24 combinations, with no exceptions -- unlike the earlier A100
// measurement of this same design, which had 3 p50-TTFT exceptions. This
// block re-derives the frontier from the report's own embedded PARETO data,
// independent of the prose, and the assertions below must be updated to
// match whatever the current run's prose actually claims -- they are not a
// fixed target every dataset must hit.
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
function assert(name, ok, detail) {
  if (!ok) fail++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ': ' + detail : ''}`);
}
assert('sole-optimal count is 24 of 24', sole === 24, `got ${sole}`);
assert('zero exceptions', exceptions.length === 0, `got ${exceptions.length}: ${JSON.stringify(exceptions)}`);
console.log(fail ? `\n${fail} FAILED` : '\nPARETO CLAIMS MATCH THE DATA');
process.exit(fail ? 1 : 0);
