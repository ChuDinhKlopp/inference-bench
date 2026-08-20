#!/usr/bin/env python
"""Build the MI250 LiveCodeBench precision report.

    python analysis/report/build_mi250_lcb_report.py

Unlike the A100 Part 1 report, these four runs have NO per-run artifacts in the
repository -- no raw/per_request.csv, no plot_data.json, no server.log. The only
surviving record is the aggregate row each run appended to
e2e_metrics_record.csv. So this report is aggregate-only: no CDFs, no engine
timeline, no batch-type analysis. Charts are static SVG (four points per metric
needs no interactivity); hover text comes from native SVG <title>.
"""

import argparse
import csv
import html
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = {  # run id -> precision arm
    "20260819_034056": "w16kv16",
    "20260819_040249": "w16kv8",
    "20260819_135325": "w8kv16",
    "20260819_154732": "w8kv8",
}
ORDER = ["w16kv16", "w16kv8", "w8kv16", "w8kv8"]
COLOR = {"w16kv16": "--s1", "w16kv8": "--s3", "w8kv16": "--s2", "w8kv8": "--s4"}
DESC = {
    "w16kv16": "BF16 weights · BF16 KV",
    "w16kv8": "BF16 weights · FP8 KV",
    "w8kv16": "FP8 weights · BF16 KV",
    "w8kv8": "FP8 weights · FP8 KV",
}


def load(csv_path):
    rows = {}
    for r in csv.DictReader(open(csv_path)):
        if r["timestamp"] in RUNS:
            rows[RUNS[r["timestamp"]]] = r
    missing = [a for a in ORDER if a not in rows]
    if missing:
        raise SystemExit(f"missing arms in {csv_path}: {missing}")
    return rows


def stat(field, key):
    """Pull one statistic out of an '(avg: X; p25: Y; ...)' cell."""
    m = re.search(rf"{key}: ([\d.]+)", field)
    return float(m.group(1)) if m else None


def kv_pool(field):
    m = re.search(r"([\d.]+) GiB ~ (\d+) tokens", field)
    return float(m.group(1)), int(m.group(2))


def derive(rows):
    d = {}
    for a in ORDER:
        r = rows[a]
        gib, toks = kv_pool(r["kv_cache_gib"])
        out = float(r["output_token_throughput_tok_s"])
        tpot = float(r["tpot_ms"])
        osl = stat(r["osl"], "avg")
        isl = stat(r["isl"], "avg")
        d[a] = dict(
            weights=float(r["model_size_gib"]), kv_gib=gib, kv_tokens=toks,
            duration=float(r["benchmark_duration_s"]), out=out,
            total=float(r["total_token_throughput_tok_s"]),
            reqs=float(r["request_throughput_req_s"]),
            ttft=float(r["ttft_ms"]), tpot=tpot, itl=float(r["itl_ms"]),
            preempt=float(r["num_preempted"]),
            kvutil=stat(r["gpu_kv_cache_usage"], "avg"),
            acc=float(r["accuracy"]), gen=float(r["total_generated_tokens"]),
            isl=isl, osl=osl,
            concurrency=out * tpot / 1000.0,          # Little's law
            eff_conc=toks / (isl + osl),              # what the cache could hold
            bytes_per_token=gib * 1024**3 / toks,
        )
    return d


# ---------------------------------------------------------------- rendering
W, H = 300, 190
ML, MR, MT, MB = 46, 14, 26, 46


def panel(title, unit, values, fmt, log=False, note=None, err=None):
    """One small-multiples bar panel: four arms, value labels, optional error bars."""
    hi = max(values[a] for a in ORDER)
    lo = min(values[a] for a in ORDER)
    iw, ih = W - ML - MR, H - MT - MB
    if log:
        import math
        l0 = math.floor(math.log10(max(min(v for v in values.values()), 1e-9)))
        l1 = math.ceil(math.log10(hi))
        span = max(l1 - l0, 1)
        y = lambda v: MT + ih - (math.log10(max(v, 10 ** l0)) - l0) / span * ih
        ticks = [10 ** e for e in range(int(l0), int(l1) + 1)]
        base = MT + ih
    else:
        # bars from zero unless the interesting variation is a narrow band
        z = 0 if lo >= 0 and lo < hi * 0.55 else lo - (hi - lo) * 0.35
        top = hi + (hi - z) * 0.18
        y = lambda v: MT + ih - (v - z) / (top - z) * ih
        ticks = [z + (top - z) * i / 3 for i in range(4)]
        base = y(z)

    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    s.append(f'<text x="0" y="11" font-size="11" font-weight="600" fill="var(--ink)" '
             f'font-family="ui-monospace,Menlo,monospace">{html.escape(title)}</text>')
    s.append(f'<text x="0" y="22" font-size="9" fill="var(--ink-3)" '
             f'font-family="ui-monospace,Menlo,monospace">{html.escape(unit)}</text>')
    for t in ticks:
        yy = y(t)
        s.append(f'<line x1="{ML}" x2="{ML+iw}" y1="{yy:.1f}" y2="{yy:.1f}" '
                 f'stroke="var(--grid)" stroke-width="1"/>')
        lab = fmt(t) if not log else (f"{t:g}")
        s.append(f'<text x="{ML-6}" y="{yy+3:.1f}" text-anchor="end" font-size="8.5" '
                 f'fill="var(--ink-3)" font-family="ui-monospace,Menlo,monospace">{lab}</text>')
    bw = iw / len(ORDER)
    for i, a in enumerate(ORDER):
        v = values[a]
        cx = ML + bw * (i + 0.5)
        yy = y(v)
        s.append(f'<rect x="{cx-bw*0.32:.1f}" y="{yy:.1f}" width="{bw*0.64:.1f}" '
                 f'height="{max(base-yy,1):.1f}" fill="var({COLOR[a]})" rx="2">'
                 f'<title>{a}: {fmt(v)}</title></rect>')
        if err:
            e = err[a]
            s.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(v-e):.1f}" y2="{y(v+e):.1f}" '
                     f'stroke="var(--ink-2)" stroke-width="1.2"/>')
        s.append(f'<text x="{cx:.1f}" y="{yy-4:.1f}" text-anchor="middle" font-size="9" '
                 f'fill="var(--ink-2)" font-family="ui-monospace,Menlo,monospace">{fmt(v)}</text>')
        s.append(f'<text x="{cx:.1f}" y="{H-30:.1f}" text-anchor="middle" font-size="8.5" '
                 f'fill="var(--ink-2)" font-family="ui-monospace,Menlo,monospace">{a}</text>')
    s.append(f'<line x1="{ML}" x2="{ML+iw}" y1="{base:.1f}" y2="{base:.1f}" '
             f'stroke="var(--rule)" stroke-width="1"/>')
    if note:
        s.append(f'<text x="{ML}" y="{H-8}" font-size="8.5" fill="var(--ink-3)" '
                 f'font-family="ui-monospace,Menlo,monospace">{html.escape(note)}</text>')
    s.append("</svg>")
    return '<div class="cell">' + "".join(s) + "</div>"


def fmt_dur(ms):
    s = ms / 1000
    if s < 90:
        return f"{s:.1f}s"
    m = s / 60
    return f"{m:.1f}m" if m < 90 else f"{m/60:.1f}h"


def table(d, rows_spec):
    h = ['<div class="tablewrap"><table><thead><tr><th>metric</th>']
    for a in ORDER:
        h.append(f"<th>{a}</th>")
    h.append("</tr></thead><tbody>")
    for label, key, fmt in rows_spec:
        h.append(f"<tr><td>{html.escape(label)}</td>")
        for a in ORDER:
            h.append(f"<td>{fmt(d[a][key])}</td>")
        h.append("</tr>")
    h.append("</tbody></table></div>")
    return "".join(h)


CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--surface-sunken:#f2f2ef;--rule:#dcdcd6;--grid:#e6e6e1;
--ink:#14161a;--ink-2:#4e5560;--ink-3:#7b828d;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#7d5bbe;
--warn-bg:#fbf4e6;--warn-ink:#6b4a06}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;
--surface:#1a1a19;--surface-sunken:#232322;--rule:#35352f;--grid:#2c2c28;--ink:#f3f3ee;--ink-2:#b9b9ad;
--ink-3:#8b8b80;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#8f6ee0;--warn-bg:#2a2416;--warn-ink:#e2c98a}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--surface-sunken:#232322;--rule:#35352f;
--grid:#2c2c28;--ink:#f3f3ee;--ink-2:#b9b9ad;--ink-3:#8b8b80;--s1:#3987e5;--s2:#d95926;--s3:#199e70;
--s4:#8f6ee0;--warn-bg:#2a2416;--warn-ink:#e2c98a}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);font-size:15px;line-height:1.55;
font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1140px;margin:0 auto;padding:40px 24px 72px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{border-bottom:1px solid var(--rule);padding-bottom:20px;margin-bottom:8px}
.eyebrow{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;letter-spacing:.1em;
text-transform:uppercase;color:var(--ink-3);margin:0 0 10px}
h1{font-size:27px;line-height:1.2;margin:0 0 8px;letter-spacing:-.015em;text-wrap:balance;font-weight:620}
.sub{margin:0;color:var(--ink-2);max-width:68ch}
section{margin-top:44px}
h2{font-size:18px;margin:0 0 4px;font-weight:600;letter-spacing:-.01em;display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
h2 .unit{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:var(--ink-3);letter-spacing:.04em;font-weight:400}
h2 .num{font-family:ui-monospace,Menlo,monospace;font-size:13px;color:var(--ink-3);font-weight:400}
.lede{margin:0 0 18px;color:var(--ink-2);max-width:72ch;font-size:14.5px}
.obs{margin:0 0 16px;padding:11px 14px 12px;border-left:3px solid var(--s1);background:var(--surface-sunken);
border-radius:0 6px 6px 0;font-size:13.5px;line-height:1.5;color:var(--ink-2);max-width:82ch}
.obs strong{color:var(--ink);font-weight:600}
.obs.kv{border-left-color:var(--s3)}.obs.warnline{border-left-color:var(--s2)}
.obs .tag{font-family:ui-monospace,Menlo,monospace;font-size:10px;letter-spacing:.07em;text-transform:uppercase;
color:var(--ink-3);display:block;margin-bottom:4px}
.obs-list{margin:6px 0;padding-left:20px;display:grid;gap:4px}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:0 0 8px}
@media (max-width:820px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:520px){.grid{grid-template-columns:1fr}}
.cell{background:var(--surface-sunken);border:1px solid var(--rule);border-radius:6px;padding:12px 12px 6px;min-width:0}
.cell svg{display:block;width:100%;height:auto;overflow:visible}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin:0 0 18px;padding:0;list-style:none}
.legend li{display:flex;align-items:center;gap:7px;font-size:13px;color:var(--ink-2)}
.legend .sw{width:22px;height:3px;border-radius:2px;flex:none}
.legend code{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;color:var(--ink)}
.tablewrap{overflow-x:auto;margin-top:14px;border:1px solid var(--rule);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:6px 11px;text-align:right;white-space:nowrap;font-family:ui-monospace,Menlo,monospace;
font-variant-numeric:tabular-nums;border-bottom:1px solid var(--grid)}
th{color:var(--ink-3);font-weight:500;background:var(--surface-sunken);position:sticky;top:0}
th:first-child,td:first-child{text-align:left}
tbody tr:last-child td{border-bottom:none}
.note{background:var(--warn-bg);color:var(--warn-ink);border-radius:5px;padding:11px 13px;font-size:13.5px;margin:18px 0 0}
.note code{font-family:ui-monospace,Menlo,monospace}
footer{margin-top:52px;padding-top:18px;border-top:1px solid var(--rule);color:var(--ink-3);font-size:12.5px}
footer code{font-family:ui-monospace,Menlo,monospace}
p code,li code,td code{font-family:ui-monospace,Menlo,monospace;font-size:12.5px}
"""


def build(d, rows):
    i = lambda v: f"{v:,.0f}"
    f1 = lambda v: f"{v:,.1f}"
    f2 = lambda v: f"{v:,.2f}"
    gib = lambda v: f"{v:,.1f}"
    tok = lambda v: (f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}k")
    pct = lambda v: f"{v:.1f}"
    acc = lambda v: f"{v:.4f}"

    base = d["w16kv16"]
    se = {a: (d[a]["acc"] * (1 - d[a]["acc"]) / 1055) ** 0.5 for a in ORDER}

    charts_mem = "".join([
        panel("model weights", "GiB (TP=4 total)", {a: d[a]["weights"] for a in ORDER}, gib),
        panel("KV cache pool", "GiB freed by FP8 weights", {a: d[a]["kv_gib"] for a in ORDER}, gib),
        panel("KV capacity", "tokens", {a: d[a]["kv_tokens"] for a in ORDER}, tok),
    ])
    charts_thr = "".join([
        panel("output throughput", "tok/s · higher better", {a: d[a]["out"] for a in ORDER}, f1),
        panel("benchmark duration", "seconds · lower better", {a: d[a]["duration"] for a in ORDER}, i),
        panel("request throughput", "req/s · higher better", {a: d[a]["reqs"] * 1000 for a in ORDER},
              lambda v: f"{v:.1f}", note="×10⁻³"),
    ])
    charts_lat = "".join([
        panel("TTFT", "ms · log · lower better", {a: d[a]["ttft"] for a in ORDER}, fmt_dur, log=True),
        panel("TPOT", "ms · lower better", {a: d[a]["tpot"] for a in ORDER}, f1),
        panel("batch width", "concurrent reqs · cap = 128", {a: d[a]["concurrency"] for a in ORDER}, f1),
    ])
    charts_sch = "".join([
        panel("preemptions", "total · lower better", {a: d[a]["preempt"] for a in ORDER}, i),
        panel("KV utilization", "% mean · 100 = cache-bound", {a: d[a]["kvutil"] for a in ORDER}, pct),
        panel("LiveCodeBench pass@1", "±1 SE (n=1055)", {a: d[a]["acc"] for a in ORDER}, acc, err=se),
    ])

    legend = "".join(
        f'<li><span class="sw" style="background:var({COLOR[a]})"></span><code>{a}</code>'
        f"<span>{DESC[a]}</span></li>" for a in ORDER)

    main_table = table(d, [
        ("model weights (GiB)", "weights", gib),
        ("KV pool (GiB)", "kv_gib", gib),
        ("KV capacity (tokens)", "kv_tokens", i),
        ("bytes per token", "bytes_per_token", f1),
        ("effective concurrency (cap ÷ ISL+OSL)", "eff_conc", f1),
        ("measured batch width (Little)", "concurrency", f1),
        ("mean KV utilization (%)", "kvutil", pct),
        ("preemptions", "preempt", i),
        ("benchmark duration (s)", "duration", i),
        ("output throughput (tok/s)", "out", f1),
        ("total token throughput (tok/s)", "total", f1),
        ("request throughput (req/s)", "reqs", lambda v: f"{v:.5f}"),
        ("TTFT (ms)", "ttft", i),
        ("TPOT (ms)", "tpot", f2),
        ("ITL (ms)", "itl", f2),
        ("generated tokens", "gen", i),
        ("LiveCodeBench pass@1", "acc", acc),
    ])

    def ratio_row(label, a, b):
        ta, tb = d[a]["tpot"], d[b]["tpot"]
        ca, cb = d[a]["concurrency"], d[b]["concurrency"]
        return (f"<tr><td>{label}</td><td>{tb/ta:.2f}×</td><td>{cb/ca:.2f}×</td>"
                f"<td>{(tb/ta)/(cb/ca):.2f}×</td></tr>")

    decomp = ("<div class=\"tablewrap\"><table><thead><tr><th>comparison</th><th>TPOT</th>"
              "<th>batch width</th><th>residual per-token cost</th></tr></thead><tbody>"
              + ratio_row("weights BF16→FP8, KV=BF16 (w16kv16→w8kv16)", "w16kv16", "w8kv16")
              + ratio_row("weights BF16→FP8, KV=FP8 (w16kv8→w8kv8)", "w16kv8", "w8kv8")
              + ratio_row("KV BF16→FP8, W=BF16 (w16kv16→w16kv8)", "w16kv16", "w16kv8")
              + ratio_row("KV BF16→FP8, W=FP8 (w8kv16→w8kv8)", "w8kv16", "w8kv8")
              + "</tbody></table></div>")

    return f"""<title>MI250 Precision Under Decode Load</title>
<style>{CSS}</style>
<div class="wrap">
<header>
  <p class="eyebrow">RIVF26 · MI250 (gfx90a/CDNA2) · LiveCodeBench release_v6 · 1,055 prompts · vLLM TRITON_ATTN</p>
  <h1>Precision on MI250: FP8 weights buy a queue-free server and cost half the throughput</h1>
  <p class="sub"><strong>Four precision arms, one decode-heavy workload.</strong> Qwen3.6-35B-A3B serves the same
    1,055 LiveCodeBench prompts at <code class="mono">max-num-seqs=128</code> under an Azure bursty arrival trace.
    Mean ISL is only <strong>{base['isl']:.0f}</strong> tokens against a mean OSL of
    <strong>{base['osl']:,.0f}</strong> — this is an extreme decode-bound regime, roughly 37 output tokens per input
    token.</p>
  <ul class="legend" style="margin-top:14px">{legend}</ul>
</header>

<div class="note"><strong>Aggregate-only report.</strong> These four runs left no per-run artifacts in the
  repository — no <code>raw/per_request.csv</code>, no <code>plot_data.json</code>, no <code>server.log</code>.
  Every number here comes from the single row each run appended to <code>e2e_metrics_record.csv</code>. There are
  therefore no latency CDFs, no engine timeline, and no batch-type analysis; TTFT/TPOT/ITL are run-level means with
  no distribution behind them.</div>

<section>
  <h2><span class="num">1</span> Memory and cache capacity</h2>
  <p class="lede">FP8 weights halve the model footprint, and on a 4×MI250 node every freed byte goes to the KV pool:
    <strong>8.14 → 34.57 GiB</strong>, a 4.25× larger cache. FP8 KV then halves the cost per token again, so capacity
    spans <strong>840k → 7.03M tokens</strong>, an 8.4× range across the four arms.</p>
  <div class="grid">{charts_mem}</div>
</section>

<section>
  <h2><span class="num">2</span> Throughput</h2>
  <p class="lede">The capacity gain does not become speed — it reverses into a large loss.</p>
  <div class="grid">{charts_thr}</div>
  <div class="obs warnline"><span class="tag">Observations</span>
    <strong>The FP8-weight arms are roughly half as fast.</strong> Output throughput falls
    <strong>680 → 319 / 297 tok/s</strong> and wall-clock rises <strong>35,175 → 71,865 / 78,689 s</strong> — the same
    1,055 prompts take <strong>2.2× longer</strong>. FP8 KV alone is nearly neutral (680 → 648 tok/s, −5%), matching
    the A100 finding that KV quantization buys residency rather than speed.</div>
</section>

<section>
  <h2><span class="num">3</span> Latency, and the queue that disappears</h2>
  <p class="lede">TTFT and TPOT move in opposite directions, by very different magnitudes. Note the log axis on TTFT.</p>
  <div class="grid">{charts_lat}</div>
  <div class="obs"><span class="tag">Observations</span>
    <ul class="obs-list">
      <li><strong>TTFT collapses by 370×</strong> — from <strong>26.2 minutes</strong> (w16kv16) to
        <strong>4.2 seconds</strong> (w8kv16). With only 840k tokens of cache and ~23k tokens per request, the
        baseline can hold about <strong>36</strong> requests; the trace delivers far more, so almost all of TTFT is
        queue wait. At 3.57M tokens the queue simply never forms.</li>
      <li><strong>TPOT moves the other way</strong>, 115 → 384 / 413 ms.</li>
      <li><strong>Batch width saturates at the scheduler cap.</strong> Measured concurrency (Little's law) is 78.5 →
        97.6 → 122.6 → 122.7 against <code class="mono">max-num-seqs=128</code>. Both FP8-weight arms are
        <strong>cap-bound, not cache-bound</strong> — their effective concurrency is 160 and 309, so most of the
        7.03M-token cache in <code class="mono">w8kv8</code> can never be spent.</li>
    </ul></div>
</section>

<section>
  <h2><span class="num">4</span> Separating the two effects</h2>
  <p class="lede">A wider batch raises TPOT on its own, so the raw 3.3× is not all dequantization cost. Dividing the
    TPOT ratio by the batch-width ratio leaves the per-token residual — and it reproduces across both KV settings.</p>
  {decomp}
  <div class="obs warnline"><span class="tag">The headline number</span>
    Switching weights BF16→FP8 costs <strong>2.13×</strong> per token at BF16 KV and <strong>2.18×</strong> at FP8 KV.
    Two independent measurements of the same quantity. gfx90a (CDNA2) has <strong>no native FP8</strong> — that
    arrives with gfx942/MI300 — so vLLM dequantizes FP8 weights into every GEMM with no arithmetic return. By
    contrast the KV-dtype residual is <strong>1.05× / 1.07×</strong>: quantizing the KV cache is close to free per
    token, and the TPOT rise it appears to cause is almost entirely the wider batch it enables.</div>
</section>

<section>
  <h2><span class="num">5</span> Scheduler pressure and accuracy</h2>
  <div class="grid">{charts_sch}</div>
  <div class="obs kv"><span class="tag">Observations</span>
    <ul class="obs-list">
      <li><strong>Preemptions go to zero.</strong> 2,781 → 1,513 → 0 → 0. The baseline thrashes badly; once the cache
        is large enough the scheduler never evicts.</li>
      <li><strong>KV utilization inverts.</strong> 95.2% → 93.2% → 52.2% → 32.1%. The first two arms are cache-bound
        and pinned near full; the FP8-weight arms leave two thirds of the pool idle because
        <code class="mono">max-num-seqs</code> binds first.</li>
      <li><strong>Accuracy is flat.</strong> pass@1 is 0.8256 / 0.8256 / 0.8237 / 0.8332 — a spread of 0.0095 against
        a binomial standard error of <strong>±0.0116</strong> on 1,055 prompts. All four arms are within one SE of one
        another; <code class="mono">w8kv8</code> scoring highest is noise, not a result.</li>
    </ul></div>
</section>

<section>
  <h2><span class="num">6</span> What this means</h2>
  <div class="obs"><span class="tag">Synthesis</span>
    On this hardware and this workload the two quantization axes do <em>different jobs</em>, and neither is simply
    "good" or "bad":
    <ul class="obs-list">
      <li><strong>FP8 KV is close to free</strong> — 2× capacity, 1.05× per-token cost, no accuracy change. Take it.</li>
      <li><strong>FP8 weights are a latency/throughput trade, not an optimization.</strong> They convert a 26-minute
        queue into a 4-second one and eliminate preemption entirely, at the price of half the throughput and 2.2×
        the wall-clock. That is worth it only if p99 TTFT is the binding requirement and total capacity is not.</li>
      <li><strong>The scheduler cap is now the bottleneck.</strong> Both FP8-weight arms sit at
        <code class="mono">max-num-seqs=128</code> with a third of the cache idle. Re-running them at a higher cap is
        the single most informative follow-up: it would show whether the throughput loss is truly the dequant path or
        partly an under-filled batch.</li>
    </ul></div>
  <div class="note"><strong>Scope.</strong> One model, one decode-heavy workload (37 output tokens per input token),
    one scheduler cap, on CDNA2. A prefill-heavy workload would weight these effects completely differently, and
    gfx942/MI300 has native FP8 which should remove the 2.1× weight penalty entirely.</div>
</section>

<section>
  <h2><span class="num">7</span> Measured values</h2>
  {main_table}
</section>

<footer>
  Source: <code>e2e_metrics_record.csv</code> on branch <code>mi250</code>, rows
  <code>20260819_034056</code> (w16kv16), <code>20260819_040249</code> (w16kv8),
  <code>20260819_135325</code> (w8kv16), <code>20260819_154732</code> (w8kv8).
  All four runs completed 1,055/1,055 requests with zero failures on <code>TRITON_ATTN</code> at
  <code>max-num-seqs=128</code> under <code>azure_multimodal_bursty_1055.csv</code>.
  Batch width is derived as output throughput × TPOT (Little's law); effective concurrency is KV capacity ÷ mean
  (ISL+OSL). Per-run telemetry was not committed, so no distributions underlie the latency means.
</footer>
</div>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=pathlib.Path, default=ROOT / "e2e_metrics_record.csv")
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "results/mi250_livecodebench_report.html")
    a = ap.parse_args()
    rows = load(a.csv)
    d = derive(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(build(d, rows))
    print(f"arms: {', '.join(ORDER)}")
    for x in ORDER:
        print(f"  {x:<8} cap {d[x]['kv_tokens']:>9,} tok  out {d[x]['out']:>7.1f} tok/s  "
              f"TPOT {d[x]['tpot']:>6.1f} ms  TTFT {fmt_dur(d[x]['ttft']):>7}  pass@1 {d[x]['acc']:.4f}")
    print(f"\nwrote {a.out}  ({a.out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
