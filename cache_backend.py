# cache_backend.py

import json
import hashlib
import io
from typing import Optional

from PIL import Image

from redis_client import get_redis

r = get_redis()
CACHE_TTL_SECONDS = 60 * 60 * 24  # 1 day


def _make_key(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True)
    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"t2i:{h}"


def get_cached_image(
    prompt: str,
    negative_prompt: str,
    model_key: str,
    height: int,
    width: int,
    steps: int,
    guidance: float,
    seed: int | None,
    sampler: str,
) -> Optional[Image.Image]:
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model_key": model_key,
        "height": height,
        "width": width,
        "steps": steps,
        "guidance": guidance,
        "seed": seed,
        "sampler": sampler,
    }
    key = _make_key(payload)
    data = r.get(key)
    if not data:
        return None

    buf = io.BytesIO(data)
    img = Image.open(buf).convert("RGB")
    return img


def set_cached_image(
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    model_key: str,
    height: int,
    width: int,
    steps: int,
    guidance: float,
    seed: int | None,
    sampler: str,
):
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model_key": model_key,
        "height": height,
        "width": width,
        "steps": steps,
        "guidance": guidance,
        "seed": seed,
        "sampler": sampler,
    }
    key = _make_key(payload)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    r.setex(key, CACHE_TTL_SECONDS, buf.getvalue())
