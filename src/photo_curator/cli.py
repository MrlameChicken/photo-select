"""Command-line interface; safe defaults, incremental indexing and local AI options."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import __version__
from .analyzer import analyze, image_paths, register_heic
from .catalog import (
    DEFAULT_DB,
    all_people,
    connect,
    photo,
    photos,
    recognized_people,
    tag_person,
    upsert_photo,
)
from .exporter import export
from .selector import choose, parse_priorities, serialize_pick


def _scan(args: argparse.Namespace) -> int:
    from .vision import check_model, evaluate

    register_heic()
    db = connect(args.db)
    matcher = None
    if args.faces:
        from .faces import FaceMatcher
        references = all_people(db)
        if not references:
            raise ValueError("No family references enrolled. Run: photo-curator enroll ./people")
        matcher = FaceMatcher(references, threshold=args.face_threshold)
    if args.vision:
        check_model(args.ollama_host, args.model)
    source = Path(args.source).expanduser().resolve()
    paths = image_paths(source)
    if args.limit is not None:
        paths = paths[:args.limit]
    stats = Counter()
    for idx, path in enumerate(paths, 1):
        previous = photo(db, path)
        file_stat = path.stat()
        same = (previous is not None and previous["size_bytes"] == file_stat.st_size
                and previous["mtime_ns"] == file_stat.st_mtime_ns)
        needs_vision = args.vision and (not same or previous["vision_model"] != args.model)
        needs_faces = matcher is not None and (not same or previous["face_revision"] != matcher.revision)
        if same and not args.force and not needs_vision and not needs_faces:
            stats["cached"] += 1
            continue
        try:
            record, image = analyze(path)
            record.update({
                "people_json": previous["people_json"] if same and previous else "[]",
                "face_confidence_json": previous["face_confidence_json"] if same and previous else "{}",
                "face_count": previous["face_count"] if same and previous else 0,
                "face_revision": previous["face_revision"] if same and previous else None,
                "category": previous["category"] if same and previous else "other",
                "caption": previous["caption"] if same and previous else "",
                "vision_score": previous["vision_score"] if same and previous else None,
                "vision_model": previous["vision_model"] if same and previous else None,
                "error": None,
            })
            if matcher:
                matched, confidences, face_count = matcher.match(image)
                record.update({
                    "people_json": json.dumps(matched),
                    "face_confidence_json": json.dumps(confidences),
                    "face_count": face_count,
                    "face_revision": matcher.revision,
                })
            if args.vision and (needs_vision or args.force):
                try:
                    record.update(evaluate(image, model=args.model, host=args.ollama_host))
                    stats["vision"] += 1
                except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                    # Failed judgments should be retried on next scan, rather than cached as success.
                    record["vision_model"] = None
                    record["vision_score"] = None
                    record["caption"] = ""
                    record["category"] = "other"
                    print(f"  Vision model failed for {path.name}: {exc}", file=sys.stderr)
                    stats["vision_failures"] += 1
            upsert_photo(db, record)
            stats["indexed"] += 1
        except (OSError, ValueError) as exc:
            print(f"  Skipped {path}: {exc}", file=sys.stderr)
            stats["errors"] += 1
        if args.verbose and idx % 25 == 0:
            print(f"Processed {idx}/{len(paths)}")
    db.close()
    print(f"Scan complete ({len(paths)} files): " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    return 0 if not stats["errors"] and not stats["vision_failures"] else 1


def _download_faces(args: argparse.Namespace) -> int:
    from .faces import download_models

    for line in download_models():
        print(line)
    print("Face detection + recognition models are now installed locally.")
    return 0


def _enroll(args: argparse.Namespace) -> int:
    from .faces import enroll_directory

    register_heic()
    db = connect(args.db)
    try:
        result = enroll_directory(db, Path(args.people_dir).expanduser().resolve())
    finally:
        db.close()
    for line in result:
        print(line)
    print("Enrolled references stored locally. Run scan --faces to match people.")
    return 0


def _tag(args: argparse.Namespace) -> int:
    db = connect(args.db)
    try:
        tag_person(db, Path(args.photo).expanduser().resolve(), args.person, remove=args.remove)
    finally:
        db.close()
    print(f"{'Removed manual label' if args.remove else 'Labeled'}: {args.person} -> {args.photo}")
    return 0


def _select(args: argparse.Namespace) -> int:
    db = connect(args.db)
    try:
        data = photos(db)
    finally:
        db.close()
    if args.source:
        source = Path(args.source).expanduser().resolve()
        if not source.is_dir():
            raise ValueError(f"Selection source directory not found: {source}")
        data = [row for row in data if Path(row["path"]).is_relative_to(source)]
    weight_parts = args.weights or []
    quota_parts = args.min_person or []
    weights = parse_priorities(weight_parts)
    quotas = {name: int(val) for name, val in parse_priorities(quota_parts, kind="quota").items()}
    picks = choose(data, args.count, weights=weights, quotas=quotas,
                   profile=args.profile, duplicate_distance=args.duplicate_distance)
    tally = Counter(p.category for p in picks)
    members = Counter(name for p in picks for name in p.people)
    print(f"Selected {len(picks)} photos; categories: {dict(tally)}; people: {dict(members)}")
    if args.dry_run:
        print(json.dumps([serialize_pick(p, i) for i, p in enumerate(picks[:20], 1)], indent=2))
        print("Dry run; no files were copied.")
    else:
        destination = Path(args.output).expanduser().resolve()
        export(picks, destination)
        print(f"Originals copied (never moved) to {destination}")
        print(f"Review the selection and reasons in {destination / 'manifest.csv'}")
    return 0


def _stats(args: argparse.Namespace) -> int:
    db = connect(args.db)
    try:
        data = photos(db)
        templates = all_people(db)
    finally:
        db.close()
    tags = Counter(name for row in data for name in recognized_people(row))
    categories = Counter(row["category"] for row in data)
    vision = sum(row["vision_score"] is not None for row in data)
    existing = sum(Path(row["path"]).is_file() for row in data)
    print(json.dumps({
        "indexed_photos": len(data), "existing_photos": existing,
        "with_vision_judgment": vision, "enrolled_people": sorted(templates),
        "identified_or_tagged": dict(tags), "categories": dict(categories),
    }, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="photo-curator", description="Offline-first family photo selection")
    p.add_argument("--version", action="version", version=__version__)
    commands = p.add_subparsers(dest="command", required=True)

    scan = commands.add_parser("scan", help="Incrementally index a local photo folder")
    scan.add_argument("source", help="Directory containing original photos (recursive)")
    scan.add_argument("--db", type=Path, default=DEFAULT_DB)
    scan.add_argument("--vision", action="store_true", help="Use local Ollama vision model")
    scan.add_argument("--model", default="gemma3:4b")
    scan.add_argument("--ollama-host", default="http://localhost:11434")
    scan.add_argument("--faces", action="store_true", help="Match enrolled reference faces locally")
    scan.add_argument("--face-threshold", type=float, default=0.50)
    scan.add_argument("--force", action="store_true", help="Reanalyze even unchanged photos")
    scan.add_argument("--limit", type=int, default=None)
    scan.add_argument("--verbose", action="store_true")
    scan.set_defaults(func=_scan)

    download = commands.add_parser("download-face-models", help="One-time download of open-licensed OpenCV face models")
    download.set_defaults(func=_download_faces)

    enroll = commands.add_parser("enroll", help="Learn family identities from local reference photos")
    enroll.add_argument("people_dir", help="Folder structure people/Mom/*.jpg, people/Dad/*.jpg")
    enroll.add_argument("--db", type=Path, default=DEFAULT_DB)
    enroll.set_defaults(func=_enroll)

    tag = commands.add_parser("tag", help="Add or remove a manual person label on an indexed photo")
    tag.add_argument("photo")
    tag.add_argument("--person", required=True)
    tag.add_argument("--db", type=Path, default=DEFAULT_DB)
    tag.add_argument("--remove", action="store_true")
    tag.set_defaults(func=_tag)

    select = commands.add_parser("select", help="Choose N photos and export copies by category")
    select.add_argument("--count", type=int, required=True)
    select.add_argument("--source", help="Restrict selection to one indexed photo folder (recursive)")
    select.add_argument("--output", required=False)
    select.add_argument("--db", type=Path, default=DEFAULT_DB)
    select.add_argument("--weights", nargs="*", default=[], metavar="NAME=WEIGHT")
    select.add_argument("--min-person", nargs="*", default=[], metavar="NAME=MINIMUM")
    select.add_argument("--profile", choices=["balanced", "family", "quality"], default="balanced")
    select.add_argument("--duplicate-distance", type=int, default=5)
    select.add_argument("--dry-run", action="store_true")
    select.set_defaults(func=_select)

    stats = commands.add_parser("stats", help="Show local index summary")
    stats.add_argument("--db", type=Path, default=DEFAULT_DB)
    stats.set_defaults(func=_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "select" and not args.dry_run and not args.output:
            raise ValueError("--output is required unless --dry-run is used")
        if args.command == "scan" and args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        if args.command == "scan" and not (0 <= args.face_threshold <= 1):
            raise ValueError("--face-threshold must be 0..1")
        return args.func(args)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
