"""SQLite-backed cache. Never stores full photo pixels or uploads images."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_DB = Path.home() / ".local-photo-curator" / "catalog.db"

PHOTO_COLUMNS = (
    "path", "size_bytes", "mtime_ns", "width", "height", "taken_at", "dhash", "sharpness",
    "exposure", "quality", "face_count", "people_json", "face_confidence_json", "category",
    "caption", "vision_score", "vision_model", "face_revision", "error", "analyzed_at",
)


def connect(db_path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=5000")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS photos (
            path TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            taken_at TEXT,
            dhash TEXT NOT NULL,
            sharpness REAL NOT NULL,
            exposure REAL NOT NULL,
            quality REAL NOT NULL,
            face_count INTEGER NOT NULL DEFAULT 0,
            people_json TEXT NOT NULL DEFAULT '[]',
            face_confidence_json TEXT NOT NULL DEFAULT '{}',
            manual_people_json TEXT NOT NULL DEFAULT '[]',
            category TEXT NOT NULL DEFAULT 'other',
            caption TEXT NOT NULL DEFAULT '',
            vision_score REAL,
            vision_model TEXT,
            face_revision TEXT,
            error TEXT,
            analyzed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS people (
            name TEXT PRIMARY KEY,
            embedding_json TEXT NOT NULL,
            reference_count INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_photos_taken_at ON photos(taken_at);
        """
    )
    db.commit()
    return db


def photo(db: sqlite3.Connection, path: Path | str) -> sqlite3.Row | None:
    return db.execute("SELECT * FROM photos WHERE path = ?", (str(Path(path).resolve()),)).fetchone()


def photos(db: sqlite3.Connection) -> list[sqlite3.Row]:
    return db.execute("SELECT * FROM photos WHERE error IS NULL ORDER BY path").fetchall()


def upsert_photo(db: sqlite3.Connection, record: dict) -> None:
    values = {**record, "analyzed_at": datetime.now(UTC).isoformat()}
    cols = ", ".join(PHOTO_COLUMNS)
    qs = ", ".join("?" for _ in PHOTO_COLUMNS)
    updates = ", ".join(f"{c}=excluded.{c}" for c in PHOTO_COLUMNS if c != "path")
    db.execute(
        f"INSERT INTO photos ({cols}) VALUES ({qs}) "
        f"ON CONFLICT(path) DO UPDATE SET {updates}",
        tuple(values.get(c) for c in PHOTO_COLUMNS),
    )
    db.commit()


def enroll_person(db: sqlite3.Connection, name: str, embedding: list[float], count: int) -> None:
    db.execute(
        """INSERT INTO people(name, embedding_json, reference_count, updated_at)
        VALUES(?,?,?,?) ON CONFLICT(name) DO UPDATE SET
        embedding_json=excluded.embedding_json, reference_count=excluded.reference_count,
        updated_at=excluded.updated_at""",
        (name, json.dumps(embedding), count, datetime.now(UTC).isoformat()),
    )
    db.commit()


def all_people(db: sqlite3.Connection) -> dict[str, list[float]]:
    rows = db.execute("SELECT name,embedding_json FROM people ORDER BY name").fetchall()
    return {r["name"]: json.loads(r["embedding_json"]) for r in rows}


def tag_person(db: sqlite3.Connection, path: Path, name: str, remove: bool = False) -> None:
    existing = photo(db, path)
    if existing is None:
        raise ValueError(f"Photo not indexed: {path}. Run scan first.")
    names = set(json.loads(existing["manual_people_json"]))
    if remove:
        names.discard(name)
    else:
        names.add(name)
    db.execute(
        "UPDATE photos SET manual_people_json = ? WHERE path = ?",
        (json.dumps(sorted(names)), str(path.resolve())),
    )
    db.commit()


def recognized_people(row: sqlite3.Row) -> set[str]:
    return set(json.loads(row["people_json"])) | set(json.loads(row["manual_people_json"]))
