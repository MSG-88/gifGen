# model_registry.py

import os
from dataclasses import dataclass, field
from typing import List, Literal, Dict

ModelKind = Literal["image", "video", "llm"]

@dataclass
class LocalModelConfig:
    key: str
    name: str
    path: str
    kind: ModelKind
    arch: str          # "sdxl", "flux", "t2v", "llm", etc.
    tags: List[str] = field(default_factory=list)
    default_height: int = 1024
    default_width: int = 1024


# 👇 adjust only if your models live somewhere else
BASE_DIR = r"M:\models"

MODEL_REGISTRY: Dict[str, LocalModelConfig] = {}


def _add(
    key: str,
    filename: str,
    kind: ModelKind,
    arch: str,
    tags: List[str],
    h: int = 1024,
    w: int = 1024,
):
    MODEL_REGISTRY[key] = LocalModelConfig(
        key=key,
        name=key,
        path=os.path.join(BASE_DIR, filename),
        kind=kind,
        arch=arch,
        tags=tags,
        default_height=h,
        default_width=w,
    )


# ===== IMAGE MODELS (SDXL / FLUX) =====

_add(
    "cyberrealistic_flux_v25",
    "cyberrealisticFlux_v25.safetensors",
    kind="image",
    arch="flux",
    tags=["photoreal", "stylized", "sfw-default"],
)

_add(
    "flesh4fantasy_v10",
    "flesh4fantasy_v10.safetensors",
    kind="image",
    arch="sdxl",
    tags=["realistic", "nsfw-leaning"],
)

_add(
    "intorealism_ultra_v50",
    "intorealismUltra_v50.safetensors",
    kind="image",
    arch="sdxl",
    tags=["photoreal", "faces", "sfw-default"],
)

_add(
    "kr345rp0_bpo",
    "kr345rp0_bpo.safetensors",
    kind="image",
    arch="sdxl",
    tags=["stylized", "sfw-default"],
)

_add(
    "morerealthanreal_v21",
    "morerealthanreal_v21.safetensors",
    kind="image",
    arch="sdxl",
    tags=["photoreal", "portraits"],
)

_add(
    "realistic_vision_v60b1_v20_novae",
    "realisticVisionV60B1_v20Novae.safetensors",
    kind="image",
    arch="sdxl",
    tags=["photoreal", "general"],
)

_add(
    "stoiqo_afrodite_flux_xl",
    "STOIQOAfroditeFLUXXL_F1DAlpha.safetensors",
    kind="image",
    arch="flux",
    tags=["stylized", "nsfw-leaning"],
)

# ===== VIDEO MODELS (PLACEHOLDERS) =====

_add(
    "krea_video_fp8",
    "kreaVideoFP8_e4m3fnScaled.safetensors",
    kind="video",
    arch="t2v",
    tags=["video", "krea"],
    h=576,
    w=1024,
)

_add(
    "smooth_mix_wan_i2v_t2v",
    "smoothMixWan22I2VT2V_t2vHighV20.safetensors",
    kind="video",
    arch="t2v",
    tags=["video", "wan", "i2v", "t2v"],
    h=576,
    w=1024,
)

# ===== LLM MODELS (PLACEHOLDERS, NOT USED IN UI YET) =====

_add(
    "mexx_qwen_tg300_23",
    "MEXX_QWEN_TG300_23.safetensors",
    kind="llm",
    arch="llm-qwen",
    tags=["qwen", "text"],
    h=0,
    w=0,
)

_add(
    "qwen_snofs_1_2_lora",
    "Qwen_Snofs_1_2.safetensors",
    kind="llm",
    arch="llm-lora",
    tags=["qwen", "lora"],
    h=0,
    w=0,
)

