import json

import pytest

from photo_curator.catalog import (
    connect,
    photo,
    photos,
    recognized_people,
    tag_person,
    upsert_photo,
)
from photo_curator.selector import choose, parse_priorities


def make_photos(tmp_path, n=8):
    db = connect(tmp_path / "db.sqlite")
    for i in range(n):
        src = tmp_path / f"photo-{i}.jpg"
        src.write_bytes(bytes([i]))
        record = {
            "path": str(src), "size_bytes": 1, "mtime_ns": 100, "width": 1200, "height": 800,
            "taken_at": f"2026:11:{i+1:02d} 10:00:00", "dhash": f"{i*100000:016x}",
            "sharpness": 0.6, "exposure": 0.8, "quality": 70 + i,
            "face_count": 1, "people_json": json.dumps(["Mom"] if i < 3 else ["Dad"]),
            "face_confidence_json": "{}", "category": "family", "caption": "", "vision_score": None,
            "vision_model": None, "face_revision": None, "error": None,
        }
        upsert_photo(db, record)
    return db


def test_quota_selects_family(tmp_path):
    db = make_photos(tmp_path)
    picks = choose(photos(db), 5, quotas={"Mom": 2}, weights={"Mom": 5}, duplicate_distance=0)
    assert len(picks) == 5
    assert len([p for p in picks if "Mom" in p.people]) >= 2
    assert len({p.path for p in picks}) == 5
    db.close()


def test_refuses_impossible_quota(tmp_path):
    db = make_photos(tmp_path)
    with pytest.raises(ValueError, match="only 3 matching"):
        choose(photos(db), 5, quotas={"Mom": 4}, duplicate_distance=0)
    db.close()


def test_duplicate_rejection(tmp_path):
    db = make_photos(tmp_path, 2)
    db.execute("UPDATE photos SET dhash='0000000000000000'")
    db.commit()
    with pytest.raises(ValueError, match="sufficiently different"):
        choose(photos(db), 2, duplicate_distance=5)
    db.close()


def test_manual_tag_and_weights(tmp_path):
    db = make_photos(tmp_path)
    path = tmp_path / "photo-7.jpg"
    tag_person(db, path, "Mom")
    assert "Mom" in recognized_people(photo(db, path))
    tag_person(db, path, "Mom", remove=True)
    assert "Mom" not in recognized_people(photo(db, path))
    assert parse_priorities(["Mom=5", "Dad=2"]) == {"Mom": 5.0, "Dad": 2.0}
    db.close()
