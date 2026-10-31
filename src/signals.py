"""Detection association and the user-approved normalized risk inputs."""
from collections import deque
import math


def associate_ppe(persons, equipment):
    """Assign each item to at most one compatible upper-body region.

    persons: dictionaries with id (int or None), box (x1,y1,x2,y2).
    equipment: dictionaries with name and box. Missing evidence is not proof
    of a violation; bare-head overrides require an observed head and no helmet.
    """
    assignments = [{"helmet": False, "head": False, "vest": False} for _ in persons]
    for item in equipment:
        name = str(item["name"]).lower().replace("-", " ").replace("_", " ")
        name = "vest" if name == "safety vest" else name
        if name not in ("helmet", "head", "vest"):
            continue
        ex1, ey1, ex2, ey2 = item["box"]
        cx, cy = (ex1 + ex2) / 2, (ey1 + ey2) / 2
        area = max(0, ex2 - ex1) * max(0, ey2 - ey1)
        if area <= 0:
            continue
        candidates = []
        for i, person in enumerate(persons):
            x1, y1, x2, y2 = person["box"]
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            lower, upper, target_y = (.2, .8, .45) if name == "vest" else (-.1, .4, .12)
            overlap = max(0, min(x2, ex2) - max(x1, ex1)) * max(0, min(y2, ey2) - max(y1, ey1))
            if x1 <= cx <= x2 and y1 + lower*h <= cy <= y1 + upper*h and overlap / area >= .5:
                distance = abs(cx - (x1+x2)/2) / w + abs(cy - (y1+target_y*h)) / h
                candidates.append((distance, i))
        if candidates:
            assignments[min(candidates)[1]][name] = True
    return [{**person, **assignment, "bare_head": assignment["head"] and not assignment["helmet"]}
            for person, assignment in zip(persons, assignments)]


def normalize_inputs(people, missing_helmet, missing_vest, crowd_count, longest_duration, safe_margin, fire_evidence):
    values = (people, missing_helmet, missing_vest, crowd_count, longest_duration, safe_margin, fire_evidence)
    if not all(math.isfinite(float(value)) for value in values) or any(value < 0 for value in values):
        raise ValueError("Signal values must be finite and nonnegative")
    if any(int(value) != value for value in values[:4]):
        raise ValueError("People and violation counts must be integers")
    if max(missing_helmet, missing_vest, crowd_count) > people or safe_margin > 1 or fire_evidence > 1:
        raise ValueError("Violation counts or normalized evidence are out of range")
    ppe = (2*missing_helmet + missing_vest) / (3*people) if people else 0.0
    zone = .5*min(crowd_count/2, 1) + .5*min(longest_duration/5, 1) if crowd_count else 0.0
    return {"ppe_load": min(ppe, 1.0), "zone_pressure": zone,
            "proximity": 1-safe_margin, "fire_evidence": fire_evidence}


class FireHistory:
    def __init__(self):
        self.checks = deque(maxlen=10)

    def add(self, confidence):
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Fire confidence must be in [0,1]")
        self.checks.append(float(confidence))

    @property
    def evidence(self):
        return max(self.checks, default=0.0)
