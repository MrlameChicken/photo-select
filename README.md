# Local Photo Curator

**Pick the best N photos from a family album without uploading anything.** A local-first Python CLI for image-quality ranking, duplicate rejection, family priorities, optional local face recognition, and optional Ollama vision judgments. It copies the chosen **originals** into categorized folders and writes a manifest explaining each choice.

> **Status:** v0.1 MVP / experimental. Quality scores are simple relative heuristics, not professional photo ratings. Face matches and model judgments are imperfect. Review the exported album before deleting anything (the app never deletes sources).

## Features in v0.1

- Recursive photo scanning: JPEG, PNG, WEBP, TIFF; HEIC via optional `pillow-heif`.
- Incremental **SQLite** cache: expensive work runs only when a file or requested AI model changes.
- Deterministic image quality metrics: sharpness, exposure, contrast, resolution; 64-bit dHash for near-duplicate rejection.
- Optional **Ollama** local vision analysis: category, caption, aesthetic/moment/expression evaluation. Ollama endpoint restricted to localhost.
- Optional **OpenCV YuNet + SFace** local face embeddings from `people/<Name>/*.jpg` reference folders, with threshold + ambiguity rejection.
- Manual person labeling as a simple fallback when you have not installed face recognition.
- Weighted selection (`Mom=5`, `Dad=4`) and minimum coverage (`Mom=20`), profile presets, light day/category diversity.
- Exactly N photos or an informative error. Originals are copied, not moved; exports are prepared in a temporary directory and published only when complete.
- JSON + CSV manifest for audit/review.

## Quick start (macOS, Linux, Python 3.11+)

```bash
git clone https://github.com/MrlameChicken/photo-select.git
cd photo-select
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# First, try a small batch. The source photos remain untouched.
photo-curator scan ~/Pictures/MyAlbum --limit 100
photo-curator stats
photo-curator select --source ~/Pictures/MyAlbum --count 20 --output ~/Pictures/Curated-20
```

The default cache is `~/.local-photo-curator/catalog.db`. Pass `--db /path/to/catalog.db` to every command to use a different index. **Don't commit your photos, face references, or catalog.** `.gitignore` ignores common local folders and databases.

### Optional: local vision AI (recommended)

Install [Ollama](https://ollama.com/download), start it, then in a terminal:

```bash
ollama pull gemma3:4b
photo-curator scan ~/Pictures/MyAlbum --vision --model gemma3:4b
# Re-running an unchanged folder will reuse cached model judgments.
photo-curator select --source ~/Pictures/MyAlbum --count 100 --profile balanced --output ~/Pictures/Curated-100
```

The model is **local**. Expect the first vision pass to be much slower than basic scanning, particularly for thousands of images. The requested model must already be installed. If you're using an Apple Silicon Mac with 48 GB of RAM, you can try a larger supported local vision model later; start with `gemma3:4b` to validate the workflow.

### Optional: identify known family members

Create *local* reference images, ideally 3–8 clear single-person photographs per person:

```text
people/
  Mom/
    mom1.jpg
    mom2.jpg
  Dad/
    dad1.jpg
  Spouse/
    spouse1.jpg
```

```bash
python -m pip install -e '.[faces]'
photo-curator download-face-models  # once: YuNet (MIT) + SFace (Apache-2.0)
photo-curator enroll ./people
photo-curator scan ~/Pictures/MyAlbum --faces --vision
photo-curator select --source ~/Pictures/MyAlbum --count 100 --profile family \
  --weights Mom=5 Dad=5 Spouse=4 \
  --min-person Mom=25 Dad=20 \
  --output ~/Pictures/Family-Highlights
```

Face recognition runs locally through OpenCV's CPU DNN runtime using **YuNet** detection and **SFace** recognition. The model downloads are explicit and stored under `~/.local-photo-curator/models/`. OpenCV Zoo lists the YuNet directory under [MIT](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and SFace directory under [Apache-2.0](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface); retain applicable third-party terms when redistributing weights. A conservative cosine threshold defaults to 0.50 and should be calibrated on your own photos.

If installing the optional face stack is difficult on Apple Silicon, you can still manually label the people in photos:

```bash
photo-curator scan ~/Pictures/MyAlbum
photo-curator tag '/absolute/path/to/photo.jpg' --person Mom
photo-curator select --source ~/Pictures/MyAlbum --count 30 --weights Mom=5 --min-person Mom=1 \
  --output ~/Pictures/Tagged-Album
```

Manual tags remain in the catalog even after rescans. `tag --remove --person Mom` removes a **manual** label (not an automatic face match). In v0.1 there is no web gallery or visual face-correction screen.

## Choosing the album

```bash
photo-curator select --source ~/Pictures/MyAlbum --count 100 --profile family \
    --weights Mom=5 Dad=5 Spouse=4 \
    --min-person Mom=20 \
    --duplicate-distance 5 \
    --dry-run

# Then export after inspecting the shortlist summary:
photo-curator select --source ~/Pictures/MyAlbum --count 100 --profile family \
    --weights Mom=5 Dad=5 Spouse=4 \
    --min-person Mom=20 \
    --output ~/Pictures/Album-100
```

| Setting | What it does |
| --- | --- |
| `--source /Photos/Trip` | Scope selections to this folder; omit to use all previously indexed photos |
| `--profile balanced` | 60% quality / 40% person importance |
| `--profile family` | 35% quality / 65% person importance |
| `--profile quality` | 85% quality / 15% person importance |
| `--weights NAME=5` | Adds importance for an identified or manually tagged person |
| `--min-person NAME=20` | Requests at least 20 selected photos showing that person |
| `--duplicate-distance 5` | Rejects candidates with dHash distance ≤5 from a selected photo; use 0 for identical-hash only |
| `--dry-run` | Lists sample picks without copying anything |

Important: the model does **not** select filenames itself. It provides structured signals. Deterministic code applies constraints, ranking and output creation. When there are not enough different photos to meet `--count` or a quota, the selection fails before writing an album. No source photo is deleted.

### Output

```text
Album-100/
  family/
    0001_b883c7aa_IMG_0012.HEIC
  portrait/
    0002_a2e1f750_IMG_0155.jpg
  travel/
    0003_d9efb999_IMG_0381.jpg
  other/
    ...
  manifest.json
  manifest.csv
```

Categories come from the local vision model when enabled, otherwise simple face-label heuristics. `family` is a *scene category*, not proof of a biological relationship. Some scenes are misclassified. The manifest records original paths, exported paths, recognized people, quality and ranking reasons. Keep exports outside your source tree so rescanning does not reindex exported copies.

## Architecture and next steps

- [Architecture and data flow](docs/ARCHITECTURE.md) — Mermaid diagrams render directly on GitHub.
- [Ranking and quality decisions](docs/SCORING.md) — formulas, tradeoffs and failure cases.
- [Next milestones](docs/ROADMAP.md) — visual review UI, better deduplication, human feedback, evaluation, performance.
- [Repository and local setup](docs/PUBLISH.md).

## Test

```bash
python -m pip install -e '.[dev]'
pytest -q
ruff check src tests
```

CI tests the deterministic core without downloading AI models. Real-world accuracy and Apple Silicon face-model installation need local acceptance tests with your own consented photo collection.

## Security and privacy

This app has no cloud backend, no analytics, and no remote photo API. Ollama HTTP calls are allowed only to a loopback host. **The explicit `download-face-models` command downloads open-licensed ONNX model weights from OpenCV Zoo on Hugging Face**; do not run enrollment until you've reviewed its terms. No family photos, embeddings or databases belong in Git. The source directory is read-only at application level; copying to any existing export folder is refused.

**Important:** This GitHub repository is currently **public**. Although no personal photos or embeddings have been committed, do not add real family images, face references or manifests to the repository. Store them separately on your local machine. You can make the repository private in GitHub settings if preferred.

## Known MVP limits

- No web UI, no browser-based preview/approval; CLI and CSV/JSON review only.
- dHash misses some near-identical bursts and may falsely group visually similar but distinct photographs.
- Image-quality heuristics can prefer textured photos over emotionally important low-light memories.
- Quality rankings are collection-dependent. The MVP does not calibrate thresholds against your preferences.
- Face recognition can misidentify people, especially low-resolution or side-facing faces. Matches require reference enrollment and conservative confidence checks.
- Overlapping minimum-person quotas use a greedy strategy. A theoretically feasible combination may still fail; no global integer optimizer yet.
- On rescanning, deleted source photos remain indexed but are excluded from selection. No catalog pruning yet.
- Cached face matches update when enrolled embeddings change; cached vision judgments update when the specified model changes. Changing model weights without changing the model tag may require `scan --force --vision`.

## License

MIT for this project's original code. Third-party model weights and libraries follow their own licenses.
