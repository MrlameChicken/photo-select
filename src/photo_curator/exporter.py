"""Copy originals into a new folder atomically; never alter the source library."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from .selector import Pick, serialize_pick


def export(picks: list[Pick], output: Path) -> list[dict]:
    destination = output.expanduser().resolve()
    if destination.exists():
        raise ValueError(f"Output directory already exists; choose a new folder: {destination}")
    original_paths = {p.path.resolve() for p in picks}
    if any(destination == p or destination in p.parents for p in original_paths):
        raise ValueError("Output cannot be a source photo path or a parent of a selected photo")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-staging-", dir=destination.parent))
    try:
        manifest: list[dict] = []
        for rank, pick in enumerate(picks, 1):
            signature = hashlib.sha256(str(pick.path).encode()).hexdigest()[:8]
            safe_name = f"{rank:04d}_{signature}_{pick.path.name}"
            folder = staging / pick.category
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pick.path, folder / safe_name)
            item = serialize_pick(pick, rank)
            item["exported"] = str(Path(pick.category) / safe_name)
            manifest.append(item)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with (staging / "manifest.csv").open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=[
                "rank", "original", "exported", "people", "category", "day", "score",
                "quality_score", "reason",
            ])
            writer.writeheader()
            for row in manifest:
                writer.writerow({**row, "people": ", ".join(row["people"])})
        if destination.exists():
            raise ValueError(f"Destination was created during export: {destination}")
        staging.rename(destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)
