import json

import numpy as np
from PIL import Image

from photo_curator.cli import main


def test_end_to_end_cli_incremental_and_non_destructive(tmp_path, capsys):
    src = tmp_path / "source"
    src.mkdir()
    rng = np.random.default_rng(12)
    originals = []
    for i in range(5):
        path = src / f"image_{i}.png"
        Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)).save(path)
        originals.append((path, path.read_bytes()))
    db = tmp_path / "db.sqlite"
    assert main(["scan", str(src), "--db", str(db)]) == 0
    assert main(["scan", str(src), "--db", str(db)]) == 0
    assert "cached=5" in capsys.readouterr().out
    assert main(["tag", str(originals[0][0]), "--person", "Mom", "--db", str(db)]) == 0
    output = tmp_path / "export"
    assert main(["select", "--db", str(db), "--count", "3", "--output", str(output),
                 "--weights", "Mom=5", "--min-person", "Mom=1", "--duplicate-distance", "0"]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest) == 3
    assert any("Mom" in item["people"] for item in manifest)
    assert (output / "manifest.csv").is_file()
    for item in manifest:
        assert (output / item["exported"]).is_file()
    for path, data in originals:
        assert path.read_bytes() == data
    # Fail safely rather than overwrite an existing album.
    assert main(["select", "--db", str(db), "--count", "3", "--output", str(output),
                 "--duplicate-distance", "0"]) == 2


def test_select_dry_run_creates_no_output(tmp_path):
    src = tmp_path / "photos"
    src.mkdir()
    db = tmp_path / "db.sqlite"
    image = src / "one.jpg"
    Image.new("RGB", (100, 100), "red").save(image)
    assert main(["scan", str(src), "--db", str(db)]) == 0
    assert main(["select", "--db", str(db), "--count", "1", "--dry-run", "--duplicate-distance", "0"]) == 0


def test_selection_can_be_scoped_to_one_album(tmp_path):
    src_a = tmp_path / "trip-a"
    src_b = tmp_path / "trip-b"
    src_a.mkdir()
    src_b.mkdir()
    db = tmp_path / "db.sqlite"
    for source in (src_a, src_b):
        Image.new("RGB", (101, 101), "orange").save(source / "picture.jpg")
        assert main(["scan", str(source), "--db", str(db)]) == 0
    folder = tmp_path / "result"
    assert main(["select", "--count", "1", "--source", str(src_a),
                 "--db", str(db), "--output", str(folder)]) == 0
    entry = json.loads((folder / "manifest.json").read_text())[0]
    assert entry["original"].startswith(str(src_a.resolve()))
