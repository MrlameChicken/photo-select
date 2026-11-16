# Roadmap — build order after MVP

## v0.1 — shipped in this starter project

CLI ingestion + SQLite cache; deterministic quality and dHash; optional Ollama; optional family reference matching; manual labels; family weights/quotas; copy originals by category; JSON/CSV manifest; tests and diagrams.

## v0.2 — highest priority: review loop and clean selection

- [ ] Local Streamlit (or FastAPI + lightweight UI) gallery: filter/sort thumbnails, compare a burst side-by-side, keep/reject, and export selected photos.
- [ ] Thumbnail caching, EXIF date display, model confidence and decision explanations.
- [ ] Person enrollment from gallery: select a few known faces, cluster unknown faces, correct false positives; explicit unknown state.
- [ ] Add negative identity labels to override a bad automatic match. Manual labels already work for positive corrections.
- [ ] Save named presets (e.g. Family album, Travel highlights) in local YAML; persist previous exports.
- [ ] Add a one-click 'replace rejected selection' operation so the count stays exact.

## v0.3 — much better deduplication and album diversity

- [ ] Group bursts by timestamp/scene, then pick best within each group (quality, eyes open, expressions).
- [ ] Add local CLIP/SigLIP visual embeddings and approximate nearest-neighbor search.
- [ ] Replace greedy quota satisfaction with integer programming (OR-Tools / CP-SAT) for hard constraints, category shares and coverage by day/event.
- [ ] Add pairwise comparison only on top candidates using a vision model; make subjective ranking cheaper.
- [ ] Incorporate face sharpness, eye openness, edge cropping and person prominence.

## v0.4 — reliable scalable library operations

- [ ] Workers with bounded CPU/RAM, durable retries, interruption/resume, HEIC performance validation on Apple Silicon.
- [ ] Schema migrations and a data retention/delete UI for biometric references.
- [ ] Metadata digest on renamed files so the cache can be reused after moves.
- [ ] Formal local eval set: person-ID precision, duplicate recall, selection accept rate, time per 1K images.
- [ ] Optional macOS native Vision / CoreML face engine to accelerate and simplify matching on Apple Silicon.
- [ ] Package a notarized macOS app with background indexing and per-album profiles.

## Acceptance criteria before calling it production-ready

- [ ] On a held-out set, very high person precision with unknown-by-default and easy corrections.
- [ ] A 100-photo export from 1,000 candidate photos is category/date diverse, with at least 90% human keep rate *after calibration* (target, not a claim).
- [ ] Scans of unchanged files avoid repeat inference; failures retry safely; interruption never damages originals.
- [ ] No image uploads or remote model calls; all cached face data can be removed by the user.
- [ ] End-to-end test with actual HEIC + JPEG on a 48-GB Apple Silicon Mac and privacy-sensitive folders.
