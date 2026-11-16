"""Deterministic selection: importance weighting, person quotas, and diversity."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .analyzer import hamming
from .catalog import recognized_people
from .vision import CATEGORIES

PROFILES = {"balanced": (0.60, 0.40), "family": (0.35, 0.65), "quality": (0.85, 0.15)}


@dataclass(frozen=True)
class Pick:
    path: Path
    people: tuple[str, ...]
    category: str
    day: str
    dhash: str
    base_score: float
    quality_score: float
    reason: str


def parse_priorities(parts: list[str], *, kind: str = "weight") -> dict[str, float]:
    parsed: dict[str, float] = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Expected Name=number, got: {part}")
        name, raw = part.rsplit("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("Priority/constraint name cannot be blank")
        try:
            number = float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid number: {part}") from exc
        if number < 0 or not math.isfinite(number):
            raise ValueError(f"Value must be a finite nonnegative number: {part}")
        if kind == "quota" and not number.is_integer():
            raise ValueError(f"Quota must be an integer: {part}")
        parsed[name] = number
    return parsed


def _candidate(row, weights: dict[str, float], profile: str) -> Pick:
    members = tuple(sorted(recognized_people(row)))
    quality = float(row["quality"])
    vision = row["vision_score"]
    if vision is not None:
        quality = 0.60 * quality + 0.40 * float(vision)
    importance = min(100.0, sum(weights.get(p, 1.0) * 15.0 for p in members))
    quality_weight, people_weight = PROFILES[profile]
    base = quality_weight * quality + people_weight * importance
    category = row["category"] if row["category"] in CATEGORIES else "other"
    if category == "other":
        if len(members) >= 2 or row["face_count"] >= 3:
            category = "family"
        elif members or row["face_count"] >= 1:
            category = "portrait"
    day = (row["taken_at"] or "unknown")[:10]
    why = f"quality={quality:.1f}; family_priority={importance:.1f}; people={','.join(members) or 'none'}"
    return Pick(Path(row["path"]), members, category, day, row["dhash"], round(base, 3), round(quality, 3), why)


def choose(
    rows: list, count: int, *, weights: dict[str, float] | None = None,
    quotas: dict[str, int] | None = None, profile: str = "balanced",
    duplicate_distance: int = 5,
) -> list[Pick]:
    if count < 1:
        raise ValueError("count must be positive")
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    if not 0 <= duplicate_distance <= 64:
        raise ValueError("duplicate_distance must be 0..64")
    weights = weights or {}
    quotas = quotas or {}
    if any(int(value) != value or value < 0 for value in quotas.values()):
        raise ValueError("Quotas must be nonnegative integers")
    candidates = [_candidate(r, weights, profile) for r in rows if Path(r["path"]).is_file()]
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} indexed photos exist; requested {count}")
    if any(n > count for n in quotas.values()):
        raise ValueError("A per-person quota exceeds the total requested photo count")
    for name, minimum in quotas.items():
        eligible = sum(name in photo.people for photo in candidates)
        if eligible < minimum:
            raise ValueError(f"Quota for {name}: need {minimum}, only {eligible} matching photos")

    chosen: list[Pick] = []
    remaining = {p.path: p for p in candidates}
    by_day: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    by_person: Counter[str] = Counter()

    def admissible(p: Pick) -> bool:
        return all(hamming(p.dhash, prior.dhash) > duplicate_distance for prior in chosen)

    def goodness(p: Pick) -> float:
        # Modest soft diversity penalty; stronger penalty applied to exact/near duplicates.
        return p.base_score - min(15.0, by_day[p.day] * 3.0) - min(10.0, by_category[p.category] * 1.5)

    def add(p: Pick) -> None:
        chosen.append(p)
        remaining.pop(p.path)
        by_day[p.day] += 1
        by_category[p.category] += 1
        by_person.update(p.people)

    # First satisfy user-requested coverage. Prefer photographs covering multiple deficits.
    while len(chosen) < count and any(by_person[name] < q for name, q in quotas.items()):
        deficits = {name for name, q in quotas.items() if by_person[name] < q}
        available = [p for p in remaining.values() if admissible(p) and deficits.intersection(p.people)]
        if not available:
            missed = {name: q - by_person[name] for name, q in quotas.items() if by_person[name] < q}
            raise ValueError(f"Cannot meet quotas without near-duplicates: {missed}. Lower quotas or duplicate-distance.")
        best = max(available, key=lambda p: (100 * len(deficits.intersection(p.people)) + goodness(p), p.base_score, str(p.path)))
        add(best)

    while len(chosen) < count:
        available = (p for p in remaining.values() if admissible(p))
        best = max(available, key=lambda p: (goodness(p), p.base_score, str(p.path)), default=None)
        if best is None:
            raise ValueError(
                f"Only {len(chosen)} sufficiently different photos could be chosen; requested {count}. "
                "Reduce --count or --duplicate-distance. No files were exported."
            )
        add(best)
    return chosen


def serialize_pick(pick: Pick, rank: int) -> dict:
    return {
        "rank": rank, "original": str(pick.path), "people": list(pick.people),
        "category": pick.category, "day": pick.day, "score": pick.base_score,
        "quality_score": pick.quality_score, "reason": pick.reason,
    }
