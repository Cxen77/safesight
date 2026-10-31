# Fuzzy rule audit

## Stage 1: 27 combinations

| PPE load | Zone pressure | Proximity | Hazard |
| --- | --- | --- | --- |
| low | low | low | low |
| low | low | medium | low |
| low | low | high | medium |
| low | medium | low | low |
| low | medium | medium | medium |
| low | medium | high | high |
| low | high | low | medium |
| low | high | medium | medium |
| low | high | high | high |
| medium | low | low | low |
| medium | low | medium | medium |
| medium | low | high | medium |
| medium | medium | low | medium |
| medium | medium | medium | medium |
| medium | medium | high | high |
| medium | high | low | medium |
| medium | high | medium | medium |
| medium | high | high | high |
| high | low | low | medium |
| high | low | medium | medium |
| high | low | high | high |
| high | medium | low | high |
| high | medium | medium | high |
| high | medium | high | high |
| high | high | low | high |
| high | high | medium | high |
| high | high | high | high |

## Stage 2: 9 combinations

| Hazard | Fire evidence | Risk |
| --- | --- | --- |
| low | low | low |
| low | medium | medium |
| low | high | high |
| medium | low | medium |
| medium | medium | high |
| medium | high | high |
| high | low | high |
| high | medium | critical |
| high | high | critical |

## Overrides

- Fire evidence >= 0.7: score 100, Critical.
- Unhelmeted intrusion >= 8 seconds: floor 85, Critical.
- Unhelmeted intrusion >= 3 seconds: floor 65, High.
- A floor never lowers a larger fuzzy score.

## Monotonic calibration

Triangular input/output membership functions, min AND, max aggregation, centroid defuzzification.
Both raw Mamdani surfaces are sampled on 21 equally spaced knots per input axis.
Axis-wise prefix maxima remove decreasing grid edges. Multilinear interpolation preserves that ordering everywhere between knots.
Runtime scores are calibrated interpolated outputs, not raw Mamdani centroid values at every continuous input.
No frozen detector test data or expert scenario labels are used for this correction.

```json
{
  "state": "passed",
  "audit": {
    "stage1_rules": 27,
    "stage2_rules": 9,
    "grid_points_per_axis": 21,
    "raw_stage1_decreasing_grid_edges": 1447,
    "raw_stage2_decreasing_grid_edges": 78,
    "stage1_knots_corrected": 3176,
    "stage2_knots_corrected": 125,
    "max_hazard_knot_correction": 0.045809832897459324,
    "max_risk_knot_correction": 1.650064606251128,
    "method": "Mamdani min-AND/max aggregation, centroid; monotonic knot correction and multilinear interpolation"
  },
  "monotonic_grid_edges_checked": 27300,
  "continuous_increase_comparisons": 10000,
  "examples": [
    {
      "inputs": [
        0,
        0,
        0,
        0,
        0
      ],
      "score": 12.361416361416364,
      "band": "Low"
    },
    {
      "inputs": [
        0.9,
        0.9,
        0.5,
        0,
        0
      ],
      "score": 62.63691493691493,
      "band": "High"
    },
    {
      "inputs": [
        0,
        0,
        0,
        0.8,
        0
      ],
      "score": 100.0,
      "band": "Critical"
    },
    {
      "inputs": [
        0.5,
        0.5,
        0.5,
        0,
        4
      ],
      "score": 65.0,
      "band": "High"
    },
    {
      "inputs": [
        0.2,
        0.1,
        0.1,
        0,
        0
      ],
      "score": 12.367137624280485,
      "band": "Low"
    }
  ],
  "scenario_agreement": "not measured; independent expert-labelled scenarios are still required"
}
```
