"""Optional local Ollama vision evaluator; image bytes never leave localhost."""
from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.parse
import urllib.request
from PIL import Image

CATEGORIES = {"family", "group", "portrait", "travel", "landscape", "food", "pet", "other"}
PROMPT = """Judge this photograph for a personal photo album. Return ONLY one JSON object with:
category: one of family,group,portrait,travel,landscape,food,pet,other;
caption: a short objective description (max 20 words);
aesthetic: integer 0..10 for composition/light;
moment: integer 0..10 for emotional or storytelling value;
expression: integer 0..10 for natural-looking expressions (5 if no people);
issues: array drawn ONLY from [closed_eyes,awkward_crop,obvious_blur,bad_exposure].
Do not guess people's identities, ages, relationships, or sensitive attributes.
If ambiguous, describe only what is visibly supported; score conservatively."""


def _local_host(host: str) -> str:
    parsed = urllib.parse.urlparse(host)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Ollama endpoint must be local http://localhost or 127.0.0.1")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("Invalid Ollama endpoint")
    return host.rstrip("/")


def _post(host: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        host + "/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def check_model(host: str, model: str) -> None:
    host = _local_host(host)
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=5) as response:
            tags = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RuntimeError("Cannot reach local Ollama. Start Ollama first.") from e
    names = {m.get("name", "") for m in tags.get("models", [])}
    if model not in names:
        raise RuntimeError(f"Model '{model}' not installed. Run: ollama pull {model}")


def _number(data: dict, key: str) -> float:
    try:
        return max(0.0, min(10.0, float(data.get(key, 5))))
    except (TypeError, ValueError):
        return 5.0


def evaluate(image: Image.Image, *, model: str = "gemma3:4b", host: str = "http://localhost:11434", timeout: int = 120) -> dict:
    host = _local_host(host)
    small = image.copy()
    small.thumbnail((960, 960), Image.Resampling.LANCZOS)
    binary = io.BytesIO()
    small.save(binary, format="JPEG", quality=80)
    raw = _post(host, {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT, "images": [base64.b64encode(binary.getvalue()).decode()]}],
        "format": "json", "stream": False, "options": {"temperature": 0},
    }, timeout)
    payload = raw.get("message", {}).get("content", "{}")
    data = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(data, dict):
        raise ValueError("Vision model returned non-object JSON")
    category = str(data.get("category", "other")).lower().strip()
    if category not in CATEGORIES:
        category = "other"
    issues_raw = data.get("issues", [])
    allowed = {"closed_eyes", "awkward_crop", "obvious_blur", "bad_exposure"}
    issues = [str(issue) for issue in issues_raw if str(issue) in allowed] if isinstance(issues_raw, list) else []
    score = 10 * (0.40 * _number(data, "aesthetic") + 0.35 * _number(data, "moment") + 0.25 * _number(data, "expression"))
    score -= 10 * len(set(issues))
    return {
        "category": category,
        "caption": str(data.get("caption", ""))[:180],
        "vision_score": round(max(0, min(100, score)), 2),
        "vision_model": model,
    }
