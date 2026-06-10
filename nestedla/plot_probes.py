"""Compare baseline-vs-nested benchmark/probe JSONs and render headline figures.

Reads from the two experiment ckpt dirs (outputs of eval_storycloze,
probe_distance, probe_continuation) and produces:

  - <out>/probe_distance.png      Δ CE (baseline - nested) per bucket, speech vs
                                  text, for both bucketings (positive = nested
                                  better; the slow-memory story predicts the
                                  speech bars grow past the fast horizon)
  - <out>/probe_continuation.png  continuation-discrimination accuracy vs
                                  context length, baseline vs nested
  - printed storycloze + summary table (markdown-ish, paste into RESULTS.md)

    python -m nestedla.plot_probes \
        --baseline /ptmp/$USER/LinearAttention/ckpt/nestedla-baseline-packed360 \
        --nested   /ptmp/$USER/LinearAttention/ckpt/nestedla-nested-packed360 \
        --out      /ptmp/$USER/LinearAttention/results
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(d: Path, name: str):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt_edges(edges):
    return [f"[{lo},{hi})" for lo, hi in zip(edges[:-1], edges[1:])]


def plot_distance(base, nest, out: Path, split: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, key, title in zip(axes, ("pos", "utt"),
                              ("by window position (context available)",
                               "by tokens since utterance start")):
        edges = base[key]["edges"]
        labels = fmt_edges(edges)
        xs = np.arange(len(labels))
        w = 0.35
        for off, mod, color in ((-w / 2, "speech", "#d62728"), (w / 2, "text", "#1f77b4")):
            d = np.array(base[key][mod]["ce"]) - np.array(nest[key][mod]["ce"])
            ax.bar(xs + off, d, w, label=f"Δ {mod}", color=color)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(xs, labels, rotation=30, ha="right")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Δ CE  (baseline − nested,  >0 ⇒ nested better)")
    axes[0].legend()
    fig.suptitle(f"Where does the nested speech gain live?  ({split}, packed)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[plot] wrote {out}")


def plot_continuation(base, nest, out: Path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for r, label, marker in ((base, "baseline (flat)", "o"), (nest, "nested", "s")):
        ns = sorted(int(k) for k in r["by_ctx"])
        ax.plot(ns, [r["by_ctx"][str(n)]["acc"] for n in ns],
                marker=marker, label=label)
    ax.axhline(0.5, color="k", lw=0.8, ls="--", label="chance")
    ax.set_xlabel("context tokens before the boundary")
    ax.set_ylabel("continuation discrimination accuracy")
    ax.set_title(f"Speech continuation vs distractor (n={base['n_points']})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[plot] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--nested", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", default="dev-clean")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- storycloze table ----
    sc_b = load(args.baseline, "eval-storycloze.json")
    sc_n = load(args.nested, "eval-storycloze.json")
    if sc_b and sc_n:
        print("\n### Spoken StoryCloze (accuracy, chance=0.5)\n")
        print("| task | baseline | nested | Δ |")
        print("|---|---:|---:|---:|")
        for task in (k for k in sc_b if isinstance(sc_b[k], dict)):
            b, n = sc_b[task]["acc_sum"], sc_n[task]["acc_sum"]
            print(f"| {task} (sum) | {b:.4f} | {n:.4f} | {n - b:+.4f} |")
            b, n = sc_b[task]["acc_mean"], sc_n[task]["acc_mean"]
            print(f"| {task} (per-token) | {b:.4f} | {n:.4f} | {n - b:+.4f} |")

    # ---- distance probe ----
    pd_b = load(args.baseline, f"probe-distance-{args.split}.json")
    pd_n = load(args.nested, f"probe-distance-{args.split}.json")
    if pd_b and pd_n:
        plot_distance(pd_b, pd_n, args.out / "probe_distance.png", args.split)
        print("\n### Δ speech CE by window position (baseline − nested)\n")
        edges = pd_b["pos"]["edges"]
        for mod in ("speech", "text"):
            d = np.array(pd_b["pos"][mod]["ce"]) - np.array(pd_n["pos"][mod]["ce"])
            cells = " ".join(f"{lab} {v:+.4f}" for lab, v in zip(fmt_edges(edges), d))
            print(f"{mod:7s} {cells}")

    # ---- continuation probe ----
    pc_b = load(args.baseline, f"probe-continuation-{args.split}.json")
    pc_n = load(args.nested, f"probe-continuation-{args.split}.json")
    if pc_b and pc_n:
        plot_continuation(pc_b, pc_n, args.out / "probe_continuation.png")
        print("\n### Continuation discrimination accuracy\n")
        print("| ctx | baseline | nested | Δ |")
        print("|---|---:|---:|---:|")
        for n in sorted(int(k) for k in pc_b["by_ctx"]):
            b, v = pc_b["by_ctx"][str(n)]["acc"], pc_n["by_ctx"][str(n)]["acc"]
            print(f"| {n} | {b:.4f} | {v:.4f} | {v - b:+.4f} |")


if __name__ == "__main__":
    main()
