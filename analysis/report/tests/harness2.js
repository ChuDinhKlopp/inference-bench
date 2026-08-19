// Exercise every control path: log/linear, clipping, smoothing, scale mode,
// arm toggling. Any throw fails the run.
const base = require('fs').readFileSync('harness.js','utf8')
  .replace(/\n\/\/ ---- assertions ----[\s\S]*$/, '\nmodule.exports = { byId, created, mkNode };\n');
require('fs').writeFileSync('harness_base.js', base);
const { byId } = require('./harness_base.js');

const fire = (id) => (byId[id].listeners.change || []).forEach((f) => f());
const isSvg = (n) => n.tag === 'svg';
const isPath = (n) => n.tag === 'path' && n.attrs.d && n.attrs.stroke && n.attrs.stroke !== 'none';
function count(node, pred, acc = { n: 0 }) {
  if (pred(node)) acc.n++;
  for (const c of node.children) count(c, pred, acc);
  return acc.n;
}
let fail = 0;
function run(name, fn, host, minPaths) {
  try {
    fn();
    const p = count(byId[host], isPath), s = count(byId[host], isSvg);
    const ok = p >= minPaths && s >= 1;
    if (!ok) fail++;
    console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}: ${s} svg, ${p} paths`);
  } catch (e) { fail++; console.log(`FAIL  ${name}: threw ${e.message}`); }
}

run('tpot log scale',      () => { byId['tpot-scale'].value = 'log';  fire('tpot-scale'); }, 'cdf-tpot', 4);
run('tpot clip p95',       () => { byId['tpot-clip'].value  = '0.95'; fire('tpot-clip');  }, 'cdf-tpot', 4);
run('tpot log + clip p99', () => { byId['tpot-clip'].value  = '0.99'; fire('tpot-clip');  }, 'cdf-tpot', 4);
run('ttft log scale',      () => { byId['ttft-scale'].value = 'log';  fire('ttft-scale'); }, 'cdf-ttft', 4);
run('ttft clip p95',       () => { byId['ttft-clip'].value  = '0.95'; fire('ttft-clip');  }, 'cdf-ttft', 4);
run('len -> osl',          () => { byId['len-metric'].value = 'osl';  fire('len-metric'); }, 'cdf-len',  4);
run('stack smooth 30s',    () => { byId['ts-smooth'].value  = '6';    fire('ts-smooth');  }, 'ts-stack', 24);
run('stack smooth 60s',    () => { byId['ts-smooth'].value  = '12';   fire('ts-smooth');  }, 'ts-stack', 24);
run('stack free scale',    () => { byId['ts-scale'].value   = 'free'; fire('ts-scale');   }, 'ts-stack', 24);

// arm toggling: click three off, then back on
const btns = byId['arm-toggles'].children.map((li) => li.children[0]);
const click = (b) => (b.listeners.click || []).forEach((f) => f());
run('one arm only (3 off)', () => { click(btns[1]); click(btns[2]); click(btns[3]); }, 'ts-stack', 6);
run('  -> cdf follows',     () => {}, 'cdf-tpot', 1);
run('last arm cannot be turned off', () => {
  click(btns[0]);
  if (btns[0].attrs['aria-pressed'] !== 'true') throw new Error('last arm was deselected');
}, 'cdf-tpot', 1);
run('pareto output-token', () => { byId['pf-thru'].value='out'; fire('pf-thru'); }, 'pareto', 0);
run('pareto request thr',  () => { byId['pf-thru'].value='req'; fire('pf-thru'); }, 'pareto', 0);
run('pareto p99',          () => { byId['pf-agg'].value='p99';  fire('pf-agg');  }, 'pareto', 0);
run('pareto p50',          () => { byId['pf-agg'].value='p50';  fire('pf-agg');  }, 'pareto', 0);
run('trace bin 0.5s', () => { byId['tr-bin'].value = '0.5'; fire('tr-bin'); }, 'trace-chart', 2);
run('trace bin 5s',   () => { byId['tr-bin'].value = '5';   fire('tr-bin'); }, 'trace-chart', 2);
run('all arms back on', () => { click(btns[1]); click(btns[2]); click(btns[3]); }, 'ts-stack', 24);

console.log(fail ? `\n${fail} FAILED` : '\nALL CONTROL PATHS OK');
process.exit(fail ? 1 : 0);
