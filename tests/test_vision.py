import json
from unittest.mock import patch

import pytest
from PIL import Image

from photo_curator.vision import evaluate


def test_vision_json_validation():
    response = {"message": {"content": json.dumps({
        "category": "family", "caption": "People smiling", "aesthetic": 8,
        "moment": 9, "expression": 8, "issues": ["awkward_crop"]
    })}}
    with patch("photo_curator.vision._post", return_value=response):
        result = evaluate(Image.new("RGB", (40, 40), "blue"))
    assert result["category"] == "family"
    assert 0 <= result["vision_score"] < 100
    assert result["vision_model"] == "gemma3:4b"


def test_vision_refuses_remote_host():
    with pytest.raises(ValueError, match="local"):
        evaluate(Image.new("RGB", (40, 40)), host="https://example.com")
