"""Two-stage Mamdani risk engine with monotonic calibrated lookup surfaces.

Both surfaces come from scikit-fuzzy min/max rules and centroid defuzzification.
Ordered rules alone do not guarantee a monotonic centroid surface. Prefix-max
correction at grid knots plus multilinear interpolation makes the deployed
mapping nondecreasing. It is a calibrated approximation, not raw centroid output
at every continuous input. Run this module to audit the correction and examples.
"""
from functools import lru_cache
from itertools import product
import json
import math

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

TERMS = ("low", "medium", "high")
INPUT_MFS = ((0, 0, .5), (.25, .5, .75), (.5, 1, 1))
RISK_MFS = ((0, 12, 25), (25, 38, 50), (50, 63, 75), (75, 88, 100))
RISK_TERMS = (*TERMS, "critical")
# Rows: hazard low/medium/high. Columns: fire low/medium/high.
STAGE2_TABLE = ((0, 1, 2), (1, 2, 2), (2, 3, 3))


def stage1_conclusion(ppe, zone, proximity):
    """Complete 27-cell ordinal rule table from the guide's detailed prompt."""
    if (sum(value == 2 for value in (ppe, zone, proximity)) >= 2
            or (ppe == 2 and zone >= 1) or (proximity == 2 and zone >= 1)):
        return 2
    if max(ppe, zone, proximity) <= 1 and sum((ppe, zone, proximity)) <= 1:
        return 0
    return 1


def validate_inputs(ppe_load, zone_pressure, proximity, fire_evidence, seconds):
    values = tuple(float(value) for value in (ppe_load, zone_pressure, proximity, fire_evidence, seconds))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Risk inputs must be finite numbers")
    if any(not 0 <= value <= 1 for value in values[:4]):
        raise ValueError("PPE, zone, proximity, and fire inputs must be between 0 and 1")
    if values[4] < 0:
        raise ValueError("Unhelmeted intrusion duration cannot be negative")
    return values


def risk_band(score):
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("Risk score must be finite and between 0 and 100")
    # Continuous scores: (25,50] is Medium, (50,75] High, (75,100] Critical.
    return "Low" if score <= 25 else "Medium" if score <= 50 else "High" if score <= 75 else "Critical"


def _variable(name, consequent=False, risk=False):
    variable_type = ctrl.Consequent if consequent else ctrl.Antecedent
    universe = np.linspace(0, 100 if risk else 1, 101)
    variable = variable_type(universe, name)
    for term, triangle in zip(RISK_TERMS if risk else TERMS, RISK_MFS if risk else INPUT_MFS):
        variable[term] = fuzz.trimf(universe, triangle)
    if consequent:
        variable.defuzzify_method = "centroid"
    return variable


def build_systems():
    ppe, zone, proximity = [_variable(name) for name in ("ppe_load", "zone_pressure", "proximity")]
    hazard = _variable("hazard", consequent=True)
    rules = []
    for p, z, x in product(range(3), repeat=3):
        conclusion = stage1_conclusion(p, z, x)
        # High: two high inputs, or high PPE/proximity together with zone >= medium.
        # Low: at most one medium input and all others low. Otherwise medium.
        rules.append(ctrl.Rule(ppe[TERMS[p]] & zone[TERMS[z]] & proximity[TERMS[x]],
                               hazard[TERMS[conclusion]], label=f"ppe {TERMS[p]}, zone {TERMS[z]}, proximity {TERMS[x]} -> {TERMS[conclusion]}"))
    hazard_input, fire = _variable("hazard"), _variable("fire_evidence")
    risk = _variable("risk", consequent=True, risk=True)
    stage2 = [ctrl.Rule(hazard_input[TERMS[h]] & fire[TERMS[f]], risk[RISK_TERMS[STAGE2_TABLE[h][f]]],
                        label=f"hazard {TERMS[h]}, fire {TERMS[f]} -> {RISK_TERMS[STAGE2_TABLE[h][f]]}")
              for h, f in product(range(3), repeat=2)]
    return ctrl.ControlSystem(rules), ctrl.ControlSystem(stage2)


def monotonic_surface(raw):
    corrected = raw.copy()
    for axis in range(raw.ndim):
        corrected = np.maximum.accumulate(corrected, axis=axis)
    return corrected


def interpolate(table, values):
    """Multilinear interpolation preserves monotonic knot ordering on every axis."""
    scaled = np.asarray(values, dtype=float) * (table.shape[0] - 1)
    lower = np.minimum(scaled.astype(int), table.shape[0] - 2)
    fraction = scaled - lower
    result = 0.0
    for corner in product((0, 1), repeat=len(values)):
        weight = np.prod([fraction[i] if step else 1 - fraction[i] for i, step in enumerate(corner)])
        result += float(table[tuple(lower[i] + step for i, step in enumerate(corner))]) * weight
    return result


class RiskEngine:
    def __init__(self, grid_points=21):
        if grid_points < 3:
            raise ValueError("Need at least three knots on each axis")
        self.grid = np.linspace(0, 1, grid_points)
        self.stage1, self.stage2 = build_systems()
        raw1 = self._surface(self.stage1, ("ppe_load", "zone_pressure", "proximity"), "hazard")
        raw2 = self._surface(self.stage2, ("hazard", "fire_evidence"), "risk")
        self.hazard_table, self.risk_table = monotonic_surface(raw1), monotonic_surface(raw2)
        self.audit = {"stage1_rules": 27, "stage2_rules": 9, "grid_points_per_axis": grid_points,
            "raw_stage1_decreasing_grid_edges": sum(int(np.count_nonzero(np.diff(raw1, axis=a) < -1e-10)) for a in range(3)),
            "raw_stage2_decreasing_grid_edges": sum(int(np.count_nonzero(np.diff(raw2, axis=a) < -1e-10)) for a in range(2)),
            "stage1_knots_corrected": int(np.count_nonzero(self.hazard_table > raw1 + 1e-10)),
            "stage2_knots_corrected": int(np.count_nonzero(self.risk_table > raw2 + 1e-10)),
            "max_hazard_knot_correction": float(np.max(self.hazard_table - raw1)),
            "max_risk_knot_correction": float(np.max(self.risk_table - raw2)),
            "method": "Mamdani min-AND/max aggregation, centroid; monotonic knot correction and multilinear interpolation"}
        self.hazard_table.setflags(write=False)
        self.risk_table.setflags(write=False)

    def _surface(self, system, names, output):
        arrays = np.meshgrid(*([self.grid] * len(names)), indexing="ij")
        simulation = ctrl.ControlSystemSimulation(system, cache=False, clip_to_bounds=False)
        for name, array in zip(names, arrays):
            simulation.input[name] = array
        simulation.compute()
        values = np.asarray(simulation.output[output], dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError("Fuzzy rules did not cover all grid inputs")
        return values

    def hazard(self, ppe_load, zone_pressure, proximity):
        values = validate_inputs(ppe_load, zone_pressure, proximity, 0, 0)
        return interpolate(self.hazard_table, values[:3])

    def fuzzy_score(self, ppe_load, zone_pressure, proximity, fire_evidence):
        values = validate_inputs(ppe_load, zone_pressure, proximity, fire_evidence, 0)
        hazard = interpolate(self.hazard_table, values[:3])
        return min(100.0, max(0.0, interpolate(self.risk_table, (hazard, values[3]))))


@lru_cache(maxsize=1)
def get_engine():
    return RiskEngine()


def compute_risk(ppe_load, zone_pressure, proximity, fire_evidence, unhelmeted_in_zone_seconds):
    ppe, zone, near, fire, seconds = validate_inputs(
        ppe_load, zone_pressure, proximity, fire_evidence, unhelmeted_in_zone_seconds)
    if fire >= .7:
        return 100.0, "Critical"
    floor = 85.0 if seconds >= 8 else 65.0 if seconds >= 3 else 0.0
    score = max(get_engine().fuzzy_score(ppe, zone, near, fire), floor)
    return score, risk_band(score)


def main():
    cases = [((0,0,0,0,0), {"Low"}), ((.9,.9,.5,0,0), {"High", "Critical"}),
             ((0,0,0,.8,0), {"Critical"}), ((.5,.5,.5,0,4), {"High", "Critical"}),
             ((.2,.1,.1,0,0), {"Low"})]
    for inputs, expected in cases:
        score, band = compute_risk(*inputs)
        print(f"{inputs} -> {score:.3f} {band}; expected {'/'.join(sorted(expected))}")
        if band not in expected:
            raise AssertionError(f"Self-test failed: {inputs}")
    print(json.dumps(get_engine().audit, indent=2))


if __name__ == "__main__":
    main()
