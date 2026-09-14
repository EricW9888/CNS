"""Render the compact public EXP-011 numerical summary figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "experiments/EXP-011-acceptance-convergence/local-metrics.json"
SPEC = ROOT / "experiments/EXP-011-acceptance-convergence/specification.json"
OUTPUT = ROOT / "figures/EXP-011-acceptance-convergence.png"


def main() -> None:
    metrics = json.loads(METRICS.read_text(encoding="utf8"))
    spec = json.loads(SPEC.read_text(encoding="utf8"))
    conditions = ["static", "yaw_positive_Z", "yaw_negative_Z",
                  "translation_positive_X", "translation_negative_X"]
    labels = ["static", "+Z", "−Z", "+X", "−X"]
    colors = ["#8c8c8c", "#1693a1", "#1693a1", "#d26837", "#d26837"]
    light = [metrics["measurements"]["combined"][c]["production_to_reference"]
             ["physical_light"]["max_abs_difference"] for c in conditions]
    stages = ["neighbor_correlators", "signed_bd", "local_emissions", "pooled_channels"]
    stage_labels = ["raw correlator", "signed stencil", "local emission", "pooled"]
    stage_values = np.asarray([
        [metrics["measurements"]["combined"][c]["production_to_reference"][stage]["relative_error"]
         for c in conditions] for stage in stages])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    ax = axes[0, 0]
    ax.bar(labels, light, color=colors)
    ax.axhline(spec["acceptance"]["maximum_physical_light_absolute_error"], color="#ad3434",
               linestyle="--", label="frozen 1e−4 gate")
    ax.set_title("Production → reference physical light")
    ax.set_ylabel("Maximum absolute light difference")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    x = np.arange(len(conditions))
    width = .19
    for i, (label, values) in enumerate(zip(stage_labels, stage_values)):
        ax.bar(x + (i - 1.5) * width, values * 100, width, label=label)
    ax.axhline(spec["acceptance"]["maximum_local_stage_relative_error"] * 100,
               color="#ad3434", linestyle="--", label="frozen 5% gate")
    ax.set_title("Final combined local stages")
    ax.set_ylabel("Relative discrepancy (%)")
    ax.set_xticks(x, labels)
    ax.legend(fontsize=7, ncols=2)

    ax = axes[1, 0]
    for condition, label, color in zip(conditions, labels, colors):
        values = metrics["measurements"]["combined"][condition]
        ax.plot([0, 1], [values["coarse_to_production"]["local_emissions"]["relative_error"] * 100,
                         values["production_to_reference"]["local_emissions"]["relative_error"] * 100],
                "o-", color=color, label=label)
    ax.axhline(5, color="#ad3434", linestyle="--")
    ax.set_xticks([0, 1], ["coarse → production", "production → reference"])
    ax.set_ylabel("Local-emission discrepancy (%)")
    ax.set_title("Refinement trend diagnostic")
    ax.legend(fontsize=7, ncols=2)

    ax = axes[1, 1]
    ax.axis("off")
    ax.text(.02, .93, "EXP-011 result", fontsize=13, weight="bold", transform=ax.transAxes)
    ax.text(.02, .78, "NEGATIVE / PARTIAL NUMERICAL RESULT", color="#ad3434",
            fontsize=11, weight="bold", transform=ax.transAxes)
    ax.text(.02, .62, "Final local stages, trends, controls, archive\n"
                      "integrity, and streaming equivalence pass.\n\n"
                      "The physical-light 1e−4 gate fails in all\n"
                      "five conditions. blocked_mass verifier failure\n"
                      "remains recorded and unelevated.",
            fontsize=10, va="top", transform=ax.transAxes)
    fig.suptitle("CNS EXP-011 · frozen full-eye optical convergence", fontsize=14)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
