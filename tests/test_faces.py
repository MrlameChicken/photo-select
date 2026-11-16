"""Mocked face backend tests: no model downloads required in CI."""
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from photo_curator.faces import FaceMatcher


class FakeDetector:
    def setInputSize(self, size):
        self.size = size

    def detect(self, image):
        return 1, np.zeros((1, 15), dtype=np.float32)


class FakeRecognizer:
    descriptor = np.array([[1.0, 0.0]], dtype=np.float32)

    def alignCrop(self, image, face):
        return image

    def feature(self, image):
        return self.descriptor


def test_conservative_face_matching(tmp_path, monkeypatch):
    (tmp_path / "face_detection_yunet_2023mar.onnx").write_bytes(b"mock")
    (tmp_path / "face_recognition_sface_2021dec.onnx").write_bytes(b"mock")
    recognizer = FakeRecognizer()
    fake_cv2 = SimpleNamespace(
        FaceDetectorYN=SimpleNamespace(create=lambda *args, **kwargs: FakeDetector()),
        FaceRecognizerSF=SimpleNamespace(create=lambda *args, **kwargs: recognizer),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    matcher = FaceMatcher({"Mom": [1, 0], "Dad": [0, 1]}, model_dir=tmp_path)
    names, confidences, count = matcher.match(Image.new("RGB", (80, 80)))
    assert names == ["Mom"] and count == 1 and confidences["Mom"] == 1.0

    recognizer.descriptor = np.array([[1.0, 1.0]], dtype=np.float32)
    names, _, count = matcher.match(Image.new("RGB", (80, 80)))
    assert names == [] and count == 1  # ambiguous scores are not assigned an ID
