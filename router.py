# router.py

from typing import Optional
from model_registry import MODEL_REGISTRY, LocalModelConfig

NSFW_KEYWORDS = {
    "nsfw", "nude", "nudity", "lingerie", "erotic", "fetish",
    "cleavage", "sexy", "sensual", "bedroom", "boudoir",
}

PORTRAIT_KEYWORDS = {
    "portrait", "close up", "headshot", "selfie", "face",
    "upper body", "bust", "model", "woman", "man",
}

LANDSCAPE_KEYWORDS = {
    "landscape", "scenery", "mountain", "forest", "cityscape",
    "street", "environment", "background",
}


def _contains_any(text: str, vocab: set) -> bool:
    text = text.lower()
    return any(word in text for word in vocab)


def auto_select_image_model(prompt: str) -> LocalModelConfig:
    p = prompt.lower()

    # NSFW-ish hints → nsfw-leaning model
    if _contains_any(p, NSFW_KEYWORDS):
        for cfg in MODEL_REGISTRY.values():
            if cfg.kind == "image" and "nsfw-leaning" in cfg.tags:
                return cfg

    # Portraits → portrait-optimized models
    if _contains_any(p, PORTRAIT_KEYWORDS):
        for key in ["morerealthanreal_v21", "intorealism_ultra_v50"]:
            if key in MODEL_REGISTRY:
                return MODEL_REGISTRY[key]

    # Landscapes / scene
    if _contains_any(p, LANDSCAPE_KEYWORDS):
        for key in ["realistic_vision_v60b1_v20_novae", "cyberrealistic_flux_v25"]:
            if key in MODEL_REGISTRY:
                return MODEL_REGISTRY[key]

    # Default fallback
    for key in ["cyberrealistic_flux_v25", "realistic_vision_v60b1_v20_novae"]:
        if key in MODEL_REGISTRY:
            return MODEL_REGISTRY[key]

    # Super fallback: first image model
    for cfg in MODEL_REGISTRY.values():
        if cfg.kind == "image":
            return cfg

    raise RuntimeError("No image models registered.")
