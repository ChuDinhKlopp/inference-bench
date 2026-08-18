// Regression guard for the tooltip-positioning bug: an absolutely-positioned
// .tip inside an unpositioned parent escapes to the document and lands nowhere
// near its chart. Every container that receives a .tip must be positioned.
const fs = require('fs');
const report = process.env.REPORT_HTML;
if (!report) { console.error('REPORT_HTML not set'); process.exit(2); }
const html = fs.readFileSync(report, 'utf8');
const css = html.split('</style>')[0];
const script = html.split('<script>')[1];

let fail = 0;
const ok = (name, cond, detail) => {
  if (!cond) fail++;
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${detail ? ': ' + detail : ''}`);
};

// .tip must be absolutely positioned (otherwise none of this matters)
ok('.tip is position:absolute', /\.tip\s*\{[^}]*position:\s*absolute/.test(css));

// Every class used as a tooltip host must carry position:relative.
// Hosts: .chartbox (CDF, Pareto) and .stack (stacked timeline, arrival trace).
for (const cls of ['chartbox', 'stack']) {
  const re = new RegExp(`\\.${cls}\\s*\\{[^}]*position:\\s*relative`);
  ok(`.${cls} is position:relative`, re.test(css));
}

// The HTML containers that JS targets by id must use one of those classes.
for (const [id, want] of [['cdf-tpot', 'chartbox'], ['cdf-ttft', 'chartbox'],
                          ['cdf-len', 'chartbox'], ['ts-stack', 'stack'],
                          ['trace-chart', 'stack']]) {
  const m = html.match(new RegExp(`<div class="([^"]*)" id="${id}"`));
  ok(`#${id} host class is positioned`, !!m && m[1].split(/\s+/).includes(want),
    m ? m[1] : 'not found');
}

// Pareto builds its own host in JS; it must set the positioned class.
ok('pareto host sets .chartbox', /box\.className\s*=\s*"chartbox"/.test(script));

console.log(fail ? `\n${fail} FAILED` : '\nTOOLTIP HOSTS ALL POSITIONED');
process.exit(fail ? 1 : 0);
