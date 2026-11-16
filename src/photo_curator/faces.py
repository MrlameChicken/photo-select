"""Opt-in local face matching using OpenCV Zoo YuNet + SFace models."""
from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from .analyzer import image_paths, load_image

MODEL_DIR = Path.home() / ".local-photo-curator" / "models"
MODEL_FILES = {
    "face_detection_yunet_2023mar.onnx": (
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        200_000,
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        30_000_000,
    ),
}


def download_models(directory: Path = MODEL_DIR) -> list[str]:
    """Explicit one-time model download; no upload and no silent downloads on inference."""
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for filename, (url, minimum_bytes) in MODEL_FILES.items():
        target = directory / filename
        if target.is_file() and target.stat().st_size >= minimum_bytes:
            results.append(f"Already present: {target}")
            continue
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temporary:
            staging = Path(temporary.name)
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    while chunk := response.read(1024 * 1024):
                        temporary.write(chunk)
                temporary.flush()
                if staging.stat().st_size < minimum_bytes:
                    raise RuntimeError(f"Invalid download for {filename}; expected ONNX model, not LFS pointer")
                staging.replace(target)
                results.append(f"Downloaded: {target}")
            finally:
                if staging.exists():
                    staging.unlink()
    return results


def _normalized(vector) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(arr)
    if norm <= 0:
        raise ValueError("Empty face embedding")
    return arr / norm


class FaceMatcher:
    def __init__(
        self, references: dict[str, list[float]], threshold: float = 0.50, margin: float = 0.06,
        model_dir: Path = MODEL_DIR,
    ):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Face recognition extra missing. Install: pip install -e '.[faces]'") from exc
        detector_model = model_dir / "face_detection_yunet_2023mar.onnx"
        recognizer_model = model_dir / "face_recognition_sface_2021dec.onnx"
        if not detector_model.is_file() or not recognizer_model.is_file():
            raise RuntimeError("Missing face model files. Run: photo-curator download-face-models")
        self.cv2 = cv2
        self.detector = cv2.FaceDetectorYN.create(str(detector_model), "", (320, 320), score_threshold=0.85)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_model), "")
        self.references = {name: _normalized(value) for name, value in references.items()}
        self.threshold = threshold
        self.margin = margin
        key = json.dumps({k: np.round(v, 6).tolist() for k, v in self.references.items()}, sort_keys=True)
        self.revision = hashlib.sha256(key.encode()).hexdigest()[:16]

    def detect(self, image: Image.Image) -> list[np.ndarray]:
        """Return normalized SFace descriptors (one per detected face)."""
        small = image.copy()
        small.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        bgr = np.asarray(small.convert("RGB"))[:, :, ::-1].copy()
        h, w = bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(bgr)
        if faces is None:
            return []
        features = []
        for face in faces:
            aligned = self.recognizer.alignCrop(bgr, face)
            features.append(_normalized(self.recognizer.feature(aligned)))
        return features

    def match(self, image: Image.Image) -> tuple[list[str], dict[str, float], int]:
        found = self.detect(image)
        names: dict[str, float] = {}
        for embedding in found:
            if not self.references:
                break
            scores = sorted(
                ((name, float(np.dot(embedding, template))) for name, template in self.references.items()),
                key=lambda pair: pair[1], reverse=True,
            )
            best, confidence = scores[0]
            runner_up = scores[1][1] if len(scores) > 1 else -1.0
            # Conservative: reject uncertain identities rather than forcing a guess.
            if confidence >= self.threshold and confidence - runner_up >= self.margin:
                names[best] = max(names.get(best, 0.0), confidence)
        return sorted(names), {name: round(score, 4) for name, score in names.items()}, len(found)


def enroll_directory(db, people_dir: Path) -> list[str]:
    from .catalog import enroll_person

    if not people_dir.is_dir():
        raise ValueError(f"People reference directory not found: {people_dir}")
    matcher = FaceMatcher({})
    results = []
    for person_dir in sorted(x for x in people_dir.iterdir() if x.is_dir()):
        reference_vectors = []
        rejected = 0
        for path in image_paths(person_dir):
            image, _, _ = load_image(path)
            detected = matcher.detect(image)
            if len(detected) != 1:
                rejected += 1
                continue
            reference_vectors.append(detected[0])
        if reference_vectors:
            centroid = _normalized(np.mean(reference_vectors, axis=0))
            enroll_person(db, person_dir.name, centroid.tolist(), len(reference_vectors))
        results.append(f"{person_dir.name}: {len(reference_vectors)} usable reference(s), {rejected} skipped")
    return results
