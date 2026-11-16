# Scoring and selection contract (MVP)

Every indexed photo receives a **deterministic quality score** between 0 and 100:

```
quality = 100 * (
    0.40 * log-normalized Laplacian edge variance
  + 0.25 * brightness proximity to midtone
  + 0.15 * normalized local contrast
  + 0.20 * minimum-dimension resolution factor
)
```

These are quick proxies for image quality, not photographic truths. They are mainly useful for relative comparisons within a similar scene. Compression and texture affect the sharpness heuristic.

When `--vision` is enabled, the vision model additionally emits `aesthetic`, `moment`, `expression` (0–10), issues and an album category. Its score is:

```
vision_score = clamp(0..100,
  10 * (0.40*aesthetic + 0.35*moment + 0.25*expression)
  - 10 * number_of_known_issues
)
photo_quality = 0.60 * deterministic_quality + 0.40 * vision_score
```

When no AI judgment is available, `photo_quality = deterministic_quality`.

For each selected photo, recognized and/or manually tagged people contribute:

```
family_importance = min(100, 15 * sum(person_weights))
base_score = quality_weight * photo_quality + people_weight * family_importance
```

Profiles: `balanced` = `0.60 / 0.40`, `family` = `0.35 / 0.65`, `quality` = `0.85 / 0.15`. Unspecified identified people default to weight 1. Photos with no enrolled/identified people have family importance 0.

The selector has 2 stages:

1. Meet `--min-person` quotas greedily, preferring photos satisfying multiple open quotas at once. Reject a proposed candidate if its dHash is within the configured Hamming distance of a previously selected photo.
2. Fill the remaining slots, selecting the highest adjusted score. Apply small per-day and per-category penalties to encourage variety. A failure to find N admissible photos returns an error before any export.

This algorithm is deterministic for unchanged catalog data and flags. But it is **not globally optimal**. In later milestones, replace greedy quotas with a constrained solver and use burst-aware visual embeddings.

## Next evaluation tests using actual family photos

Measure Precision@K with manual keep/reject labels; person identification precision (prefer high precision over recall); false duplicate rejections; category accuracy; shortlist coverage across dates and family members; and user-rejected fraction after review. Include a protected test split by event/date to avoid tuning on the same pictures used for evaluation. Start with 100–200 representative photos before processing tens of thousands.
