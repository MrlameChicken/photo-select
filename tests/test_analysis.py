import numpy as np
from PIL import Image
from photo_curator.analyzer import analyze, dhash, hamming, metrics


def test_dhash_stable_and_64bit():
    im = Image.fromarray(np.arange(100, dtype=np.uint8).reshape(10, 10))
    assert dhash(im) == dhash(im.copy())
    assert len(dhash(im)) == 16
    assert hamming(dhash(im), dhash(im.copy())) == 0


def test_better_contrast_or_sharpness_than_flat_image():
    flat = Image.fromarray(np.full((120, 120), 128, dtype=np.uint8))
    arr = np.indices((120, 120)).sum(axis=0) % 2 * 255
    textured = Image.fromarray(arr.astype(np.uint8))
    score_a = metrics(flat, 120, 120)["quality"]
    score_b = metrics(textured, 120, 120)["quality"]
    assert score_b > score_a


def test_analyze_produces_metadata(tmp_path):
    source = tmp_path / "one.jpg"
    Image.new("RGB", (80, 60), "orange").save(source)
    row, image = analyze(source)
    assert row["width"] == 80 and row["height"] == 60
    assert row["size_bytes"] > 0
    assert row["path"] == str(source.resolve())
    assert image.size == (80, 60)
