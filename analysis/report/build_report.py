#!/usr/bin/env python
"""Build the RIVF26 Part 1 precision performance report.

Reads the per-run artifacts of the four precision arms, extracts the compact
series the report's charts need, and injects them into template.html to produce
a single self-contained HTML file with no runtime dependencies.

    python analysis/report/build_report.py

Run directories are discovered by precision suffix; the newest match wins. Pin
one explicitly with --run, e.g.:

    --run w8kv16=results/part1/performance/20260817_171227_..._w8kv16_mns256_bin500ms

Only three files per arm are read (~1.7 MB of a ~248 MB run directory):

    <run>/raw/per_request.csv    -> TTFT / TPOT / ISL / OSL distributions
    <run>/plot_data.json         -> stacked engine timeline
    <run>/<run_id>_e2e_record_input.json  -> aggregate check (see NOTE below)

plus, once, the Azure arrival trace CSV.

NOTE: the Pareto chart (5.1) and the numeric tables in the prose are currently
hand-written constants inside template.html, NOT generated here. --check-pareto
compares the template's constants against the measured runs and fails loudly if
they have drifted, so a rebuild on different runs cannot silently publish stale
numbers.
"""

import argparse
import csv
import json
import pathlib
import sys

PRECISIONS = ["w16kv16", "w8kv16", "w16kv8", "w8kv8"]
RUN_GLOB = "*_performance_pubmed_1000_longest_{p}_mns256_bin500ms"
BIN_FACTOR = 10  # plot_data.json is 500 ms binned; the browser gets 5 s bins

# rivf26/analysis/report/build_report.py -> rivf26/
ROOT = pathlib.Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
def discover(results_dir, overrides):
    """Map each precision to a run directory, newest match winning."""
    runs = {}
    for p in PRECISIONS:
        if p in overrides:
            d = pathlib.Path(overrides[p])
            if not d.is_absolute():
                d = ROOT / d
        else:
            found = sorted(results_dir.glob(RUN_GLOB.format(p=p)))
            if not found:
                sys.exit(f"no run directory for {p} under {results_dir}\n"
                         f"  expected something matching {RUN_GLOB.format(p=p)}")
            d = found[-1]
        if not d.is_dir():
            sys.exit(f"{p}: not a directory: {d}")
        runs[p] = d
    return runs


def per_request_csv(run_dir):
    """raw/ is a symlink into bulk storage; fail with a useful message if it dangles."""
    f = run_dir / "raw" / "per_request.csv"
    if not f.exists():
        link = run_dir / "raw"
        target = f"  ({link} -> {link.resolve()})" if link.is_symlink() else ""
        sys.exit(f"missing {f}{target}\n"
                 f"  raw/ is a symlink into RIVF26_BULK_ROOT. If this checkout was\n"
                 f"  copied from another machine, re-copy with `cp -L`/`rsync -L`,\n"
                 f"  or pass --run {run_dir.name.split('_')[-3]}=<dir> for a local run.")
    return f


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------
def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def agg(series, how, nd):
    """Down-bin a 500 ms series, one aggregate per BIN_FACTOR-wide window."""
    out = []
    for w in chunks(series, BIN_FACTOR):
        w = [v for v in w if v is not None]
        if not w:
            out.append(0)
            continue
        v = {"mean": sum(w) / len(w), "max": max(w), "min": min(w), "sum": sum(w)}[how]
        out.append(int(v) if nd == 0 else round(v, nd))
    return out


def extract_cdf(run_dir):
    rows = list(csv.DictReader(open(per_request_csv(run_dir))))
    tpot = sorted(float(r["tpot_s"]) * 1000.0 for r in rows if float(r["tpot_s"]) > 0)
    return {
        "ttft": [round(float(r), 1) for r in sorted(float(x["ttft_s"]) for x in rows)],
        "tpot": [round(v, 1) for v in tpot],
        "isl": sorted(int(r["prompt_len"]) for r in rows),
        "osl": sorted(int(r["output_tokens"]) for r in rows),
    }


def extract_ts(run_dir):
    pd = json.load(open(run_dir / "plot_data.json"))
    entry = list(pd["TS"].values())[0]
    if entry["bin_s"] != 0.5:
        sys.exit(f"{run_dir.name}: expected 0.5 s bins, got {entry['bin_s']}")
    r = entry["runs"][run_dir.name]
    return {
        "cap": r["cap"],
        "bin_s": 0.5 * BIN_FACTOR,
        "kv": agg(r["kv"], "mean", 4),        # fraction 0..1
        "hbm": agg(r["hbm"], "mean", 2),      # percent
        "run": agg(r["run"], "mean", 1),
        "runmin": agg(r["run"], "min", 1),    # preserves preemption churn
        "wait": agg(r["wait"], "mean", 1),
        "pre": agg(r["pre"], "max", 0),       # cumulative, monotonic
        "arr": agg(r["arrivals"], "sum", 0),
    }


def extract_trace(trace_csv):
    rows = list(csv.DictReader(open(trace_csv)))
    off = [float(r["ARRIVAL_OFFSET_S"]) for r in rows]
    return {"off": [round(v, 3) for v in off], "dur": round(off[-1], 3)}


def serving_metrics(run_dir):
    """Aggregates recorded by bench.py, used only to audit the template constants."""
    f = next(run_dir.glob("*_e2e_record_input.json"), None)
    if f is None:
        return None
    return json.load(open(f))["serving_metrics"]


# --------------------------------------------------------------------------
# the hand-written-constant audit
# --------------------------------------------------------------------------
def check_pareto(template, runs):
    """Compare template.html's PARETO literals against the measured runs.

    The Pareto chart is the one plot whose data is typed into the template
    rather than generated. Rebuilding on different runs would leave it showing
    the old numbers while every other chart updated -- it renders fine and is
    wrong, which is worse than a crash. So verify it explicitly.
    """
    src = template[template.index("const PARETO = {"):]
    src = src[:src.index("\n  };") + 5]
    body = src[src.index("{"):src.rindex("}") + 1]
    # JS object literal with bare keys -> JSON
    import re
    baked = json.loads(re.sub(r"(\w+):", r'"\1":', body).replace(",\n  }", "\n  }"))

    pct = {50: "p50", 90: "p90", 99: "p99"}
    drift = []
    for p, d in runs.items():
        m = serving_metrics(d)
        if m is None:
            continue
        want = {
            "tot": m["total_token_throughput"], "out": m["output_throughput"],
            "req": m["request_throughput"],
            "ttft": {"mean": m["mean_ttft_ms"] / 1000.0},
            "tpot": {"mean": m["mean_tpot_ms"]},
        }
        for q, v in m["percentiles_ttft_ms"]:
            if int(q) in pct:
                want["ttft"][pct[int(q)]] = v / 1000.0
        for q, v in m["percentiles_tpot_ms"]:
            if int(q) in pct:
                want["tpot"][pct[int(q)]] = v
        got = baked.get(p)
        if got is None:
            drift.append(f"{p}: absent from template PARETO")
            continue
        for k in ("tot", "out", "req"):
            if abs(got[k] - want[k]) > max(abs(want[k]) * 0.002, 1e-4):
                drift.append(f"{p}.{k}: template {got[k]} vs measured {want[k]:.4f}")
        for met in ("ttft", "tpot"):
            for stat, v in want[met].items():
                g = got[met][stat]
                if abs(g - v) > max(abs(v) * 0.002, 0.05):
                    drift.append(f"{p}.{met}.{stat}: template {g} vs measured {v:.1f}")
    return drift


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=pathlib.Path,
                    default=ROOT / "results/part1/performance")
    ap.add_argument("--trace", type=pathlib.Path,
                    default=ROOT / "traces/processed/azure_multimodal_bursty_1000.csv")
    ap.add_argument("--template", type=pathlib.Path,
                    default=ROOT / "analysis/report/template.html")
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "results/part1/performance/precision_performance_report.html")
    ap.add_argument("--run", action="append", default=[], metavar="PRECISION=DIR",
                    help="pin one arm's run directory instead of discovering it")
    ap.add_argument("--emit-json", type=pathlib.Path,
                    help="also write cdf/ts/trace JSON here (for debugging)")
    ap.add_argument("--no-check-pareto", action="store_true",
                    help="skip the hand-written-constant audit (not recommended)")
    args = ap.parse_args()

    overrides = {}
    for spec in args.run:
        if "=" not in spec:
            sys.exit(f"--run expects PRECISION=DIR, got {spec!r}")
        k, v = spec.split("=", 1)
        if k not in PRECISIONS:
            sys.exit(f"unknown precision {k!r}; expected one of {PRECISIONS}")
        overrides[k] = v

    runs = discover(args.results_dir, overrides)
    print("run directories")
    for p in PRECISIONS:
        print(f"  {p:<8} {runs[p].name}")

    cdf = {p: extract_cdf(d) for p, d in runs.items()}
    ts = {p: extract_ts(d) for p, d in runs.items()}
    trace = extract_trace(args.trace)

    print("\nextracted")
    for p in PRECISIONS:
        print(f"  {p:<8} {len(cdf[p]['ttft']):>5} requests   "
              f"{len(ts[p]['kv']):>5} bins @ {ts[p]['bin_s']:g}s   "
              f"cap {ts[p]['cap']:>9,} tok")
    print(f"  trace    {len(trace['off']):>5} arrivals   {trace['dur']:.1f}s window")

    template = args.template.read_text()

    if not args.no_check_pareto:
        drift = check_pareto(template, runs)
        if drift:
            print("\nFAIL: template.html's hand-written PARETO constants no longer "
                  "match the measured runs:", file=sys.stderr)
            for d in drift:
                print(f"  {d}", file=sys.stderr)
            print("\nThe Pareto chart (5.1) and the numeric tables in the prose are "
                  "not generated from data.\nUpdate them in template.html, or pass "
                  "--no-check-pareto to build anyway.", file=sys.stderr)
            return 1
        print("\npareto constants: match measured runs")

    dumps = lambda o: json.dumps(o, separators=(",", ":"))
    html = template
    for token, payload in (("__CDF__", cdf), ("__TS__", ts), ("__TRACE__", trace)):
        if token not in html:
            sys.exit(f"template is missing the {token} placeholder")
        html = html.replace(token, dumps(payload))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"\nwrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")

    if args.emit_json:
        args.emit_json.mkdir(parents=True, exist_ok=True)
        for name, payload in (("cdf", cdf), ("ts", ts), ("trace", trace)):
            (args.emit_json / f"{name}.json").write_text(dumps(payload))
        print(f"wrote cdf/ts/trace JSON to {args.emit_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
