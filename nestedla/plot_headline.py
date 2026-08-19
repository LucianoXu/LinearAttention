"""Headline figure: baseline vs nested, per-modality validation perplexity.

The project's hypothesis is that adding a slower second memory level helps
*speech* tokens (long, dense unit spans) more than *text* tokens. This script
reads the per-modality validation losses that train.py logs to stdout --

    [step 200] valid: loss=3.1 text=2.4 speech=3.8 special=0.5

-- for the baseline and nested runs, converts CE -> perplexity, and produces:
  (1) validation-PPL curves over training, per modality, baseline vs nested;
  (2) a grouped bar chart of the final per-modality PPL;
and prints the headline table (Delta_speech should exceed Delta_text).

It parses logs (no GPU, no checkpoint needed -- survives the cluster being
down). If --baseline-json/--nested-json are given (from nestedla.eval) those
full-split numbers are used for the bar chart / table instead of the last
in-training validation point.

    python -m nestedla.plot_headline \
        --baseline-log /ptmp/$USER/LinearAttention/logs/nestedla-<jobA>.out \
        --nested-log   /ptmp/$USER/LinearAttention/logs/nestedla-<jobB>.out \
        --out          /ptmp/$USER/LinearAttention/results/headline.png
"""

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODS = ["text", "speech", "special"]
_VALID_RE = re.compile(
    r"\[step (\d+)\] valid:\s*loss=([\d.]+)\s+text=([\d.]+)\s+speech=([\d.]+)\s+special=([\d.]+)")


def parse_log(path):
    """Return {'step': [...], 'loss': [...], 'text': [...], ...} of CE values."""
    series = {k: [] for k in ("step", "loss", *MODS)}
    text = Path(path).read_text(errors="ignore")
    for m in _VALID_RE.finditer(text):
        step, loss, t, s, sp = m.groups()
        series["step"].append(int(step))
        series["loss"].append(float(loss))
        series["text"].append(float(t))
        series["speech"].append(float(s))
        series["special"].append(float(sp))
    return series


def _ppl(ce):
    return math.exp(ce) if ce < 50 else float("inf")


def _autofind(logs_dir, experiment):
    """Newest <logs_dir>/nestedla-*.out whose body names the experiment."""
    cands = sorted(glob.glob(os.path.join(logs_dir, "nestedla-*.out")),
                   key=os.path.getmtime, reverse=True)
    for c in cands:
        body = Path(c).read_text(errors="ignore")
        if f"experiment_name={experiment}" in body or experiment in body:
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description="Headline baseline-vs-nested per-modality PPL plot")
    user = os.environ.get("USER", "yinxu")
    logs_dir = f"/ptmp/{user}/LinearAttention/logs"
    ap.add_argument("--baseline-log", default=None)
    ap.add_argument("--nested-log", default=None)
    ap.add_argument("--logs-dir", default=logs_dir)
    ap.add_argument("--baseline-json", default=None, help="optional nestedla.eval JSON")
    ap.add_argument("--nested-json", default=None)
    ap.add_argument("--out", default=f"/ptmp/{user}/LinearAttention/results/headline.png")
    args = ap.parse_args()

    bl_log = args.baseline_log or _autofind(args.logs_dir, "nestedla-baseline")
    ns_log = args.nested_log or _autofind(args.logs_dir, "nestedla-nested")
    if not bl_log or not ns_log:
        raise SystemExit(f"could not locate logs (baseline={bl_log}, nested={ns_log}); "
                         f"pass --baseline-log/--nested-log explicitly")
    print(f"[plot] baseline log: {bl_log}\n[plot] nested   log: {ns_log}")

    bl, ns = parse_log(bl_log), parse_log(ns_log)
    if not bl["step"] or not ns["step"]:
        raise SystemExit("no '[step N] valid:' lines parsed yet -- runs may not have validated")

    # final per-modality PPL: prefer eval JSON, else last in-training validation
    def final_ppl(series, json_path):
        if json_path and Path(json_path).exists():
            d = json.loads(Path(json_path).read_text())
            return {m: d[f"ppl_{m}"] for m in MODS}, f"eval JSON (step {d.get('step')})"
        return {m: _ppl(series[m][-1]) for m in MODS}, f"in-training valid (step {series['step'][-1]})"

    bl_final, bl_src = final_ppl(bl, args.baseline_json)
    ns_final, ns_src = final_ppl(ns, args.nested_json)

    # ---- figure ----
    fig, (axc, axb) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"text": "#1f77b4", "speech": "#d62728", "special": "#2ca02c"}
    for m in MODS:
        axc.plot(bl["step"], [_ppl(c) for c in bl[m]], color=colors[m],
                 ls="-", label=f"{m} (baseline)")
        axc.plot(ns["step"], [_ppl(c) for c in ns[m]], color=colors[m],
                 ls="--", label=f"{m} (nested)")
    axc.set_xlabel("step"); axc.set_ylabel("validation perplexity")
    axc.set_title("Per-modality validation PPL\nsolid = baseline, dashed = nested")
    axc.legend(fontsize=8); axc.grid(alpha=0.3)

    x = range(len(MODS)); w = 0.38
    axb.bar([i - w / 2 for i in x], [bl_final[m] for m in MODS], w, label="baseline", color="#888")
    axb.bar([i + w / 2 for i in x], [ns_final[m] for m in MODS], w, label="nested", color="#d62728")
    axb.set_xticks(list(x)); axb.set_xticklabels(MODS)
    axb.set_ylabel("final validation perplexity")
    axb.set_title("Final per-modality PPL"); axb.legend(); axb.grid(alpha=0.3, axis="y")
    for i, m in enumerate(MODS):
        axb.text(i + w / 2, ns_final[m], f"{ns_final[m]:.2f}", ha="center", va="bottom", fontsize=8)
        axb.text(i - w / 2, bl_final[m], f"{bl_final[m]:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"[plot] saved {out}")

    # ---- headline table ----
    print(f"\n=== HEADLINE: baseline vs nested per-modality PPL ===")
    print(f"baseline final from: {bl_src}\nnested   final from: {ns_src}")
    print(f"{'modality':<9} {'baseline':>10} {'nested':>10} {'delta':>9} {'improve%':>9}")
    deltas = {}
    for m in MODS:
        b, n = bl_final[m], ns_final[m]
        d = b - n
        deltas[m] = d
        pct = 100.0 * d / b if b else 0.0
        print(f"{m:<9} {b:>10.3f} {n:>10.3f} {d:>9.3f} {pct:>8.2f}%")
    # honest verdict: the hypothesis is "nested lowers speech PPL, and more so
    # than text". That needs the speech delta to be positive (nested better) AND
    # larger than the text delta. A negative speech delta means nested is *worse*
    # on speech -- not support, regardless of how text moved.
    ds_sp, ds_tx = deltas["speech"], deltas["text"]
    if ds_sp > 0 and ds_sp > ds_tx:
        verdict = "nested lowers speech PPL more than text -> hypothesis SUPPORTED"
    elif ds_sp <= 0:
        verdict = "nested does NOT lower speech PPL (delta<=0) -> hypothesis NOT supported"
    else:
        verdict = "nested helps text >= speech -> hypothesis NOT supported"
    print(f"\nDelta_speech={ds_sp:+.3f}  Delta_text={ds_tx:+.3f}  (delta = baseline - nested; >0 means nested better)")
    print(f"=> {verdict}")


if __name__ == "__main__":
    main()
