"""Audit the rule tables and monotonic engine without detector/test-data inputs."""
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from risk import compute_risk, get_engine, stage1_conclusion, STAGE2_TABLE, TERMS, RISK_TERMS


def main():
    engine = get_engine()
    edges = 0
    for table in (engine.hazard_table, engine.risk_table):
        for axis in range(table.ndim):
            differences = np.diff(table, axis=axis)
            if np.any(differences < -1e-10):
                raise AssertionError("A calibrated grid edge is decreasing")
            edges += differences.size
    rng, comparisons = np.random.default_rng(42), 0
    for _ in range(2000):
        inputs = [*rng.random(4), float(rng.uniform(0, 12))]
        baseline = compute_risk(*inputs)[0]
        for axis in range(5):
            higher = inputs.copy()
            ceiling = 12 if axis == 4 else 1
            higher[axis] += (ceiling - higher[axis]) * rng.random()
            if compute_risk(*higher)[0] < baseline - 1e-9:
                raise AssertionError(f"Risk decreased: {inputs}, {higher}")
            comparisons += 1
    examples = [[0,0,0,0,0], [.9,.9,.5,0,0], [0,0,0,.8,0], [.5,.5,.5,0,4], [.2,.1,.1,0,0]]
    report = {"state": "passed", "audit": engine.audit, "monotonic_grid_edges_checked": edges,
              "continuous_increase_comparisons": comparisons,
              "examples": [{"inputs": inputs, "score": compute_risk(*inputs)[0],
                            "band": compute_risk(*inputs)[1]} for inputs in examples],
              "scenario_agreement": "not measured; independent expert-labelled scenarios are still required"}
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "risk-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Fuzzy rule audit", "", "## Stage 1: 27 combinations", "",
             "| PPE load | Zone pressure | Proximity | Hazard |", "| --- | --- | --- | --- |"]
    for p, z, x in product(range(3), repeat=3):
        lines.append(f"| {TERMS[p]} | {TERMS[z]} | {TERMS[x]} | {TERMS[stage1_conclusion(p,z,x)]} |")
    lines += ["", "## Stage 2: 9 combinations", "", "| Hazard | Fire evidence | Risk |", "| --- | --- | --- |"]
    for h, f in product(range(3), repeat=2):
        lines.append(f"| {TERMS[h]} | {TERMS[f]} | {RISK_TERMS[STAGE2_TABLE[h][f]]} |")
    lines += ["", "## Overrides", "", "- Fire evidence >= 0.7: score 100, Critical.",
              "- Unhelmeted intrusion >= 8 seconds: floor 85, Critical.",
              "- Unhelmeted intrusion >= 3 seconds: floor 65, High.",
              "- A floor never lowers a larger fuzzy score.", "", "## Monotonic calibration", "",
              "Triangular input/output membership functions, min AND, max aggregation, centroid defuzzification.",
              "Both raw Mamdani surfaces are sampled on 21 equally spaced knots per input axis.",
              "Axis-wise prefix maxima remove decreasing grid edges. Multilinear interpolation preserves that ordering everywhere between knots.",
              "Runtime scores are calibrated interpolated outputs, not raw Mamdani centroid values at every continuous input.",
              "No frozen detector test data or expert scenario labels are used for this correction.", "", "```json",
              json.dumps(report, indent=2), "```"]
    (results / "risk_rules.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
