# api_server.py

import base64
import io
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from model_manager import ModelManager
from model_registry import MODEL_REGISTRY, LocalModelConfig
from router import auto_select_image_model
from cache_backend import get_cached_image, set_cached_image
from vector_store import init_schema, log_generation

QUALITY_BOOST = "ultra-detailed, sharp focus, high dynamic range, 8k, photorealistic textures, crisp edges"
NEGATIVE_CLEANUP = "lowres, watermark, text, logo, oversaturated, deformed, extra limbs, artifacts, noise"

app = FastAPI(title="Local Diffusion API", version="1.0")

# Initialize heavy resources once
_MODEL_MANAGER = ModelManager()
init_schema()


def _build_prompts(prompt: str, negative_prompt: str, apply_quality_boost: bool, apply_artifact_cleanup: bool) -> tuple[str, str]:
    final_prompt = prompt.strip()
    if apply_quality_boost and final_prompt:
        final_prompt = f"{final_prompt}, {QUALITY_BOOST}"

    final_negative = negative_prompt.strip()
    if apply_artifact_cleanup:
        final_negative = (
            f"{final_negative}, {NEGATIVE_CLEANUP}"
            if final_negative
            else NEGATIVE_CLEANUP
        )
    return final_prompt, final_negative


def _encode_image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Main prompt / storyline")
    negative_prompt: str = Field("", description="Negative prompt")
    model_key: Optional[str] = Field(None, description="Explicit model key; omit for auto selection")
    height: int = Field(1024, ge=256, le=1536)
    width: int = Field(1024, ge=256, le=1536)
    steps: int = Field(28, ge=5, le=80)
    guidance: float = Field(5.5, ge=1.0, le=20.0)
    seed: Optional[int] = Field(None, ge=0)
    sampler: str = Field("dpmpp_2m_karras", description="Sampler id (matches UI options)")
    apply_quality_boost: bool = True
    apply_artifact_cleanup: bool = True


class GenerateResponse(BaseModel):
    model_key: str
    prompt: str
    negative_prompt: str
    height: int
    width: int
    steps: int
    guidance: float
    seed: Optional[int]
    sampler: str
    cached: bool
    image_base64: str


class ModelInfo(BaseModel):
    key: str
    name: str
    kind: str
    arch: str
    tags: List[str]
    default_height: int
    default_width: int


@app.get("/models", response_model=List[ModelInfo])
def list_models():
    return [
        ModelInfo(
            key=cfg.key,
            name=cfg.name,
            kind=cfg.kind,
            arch=cfg.arch,
            tags=cfg.tags,
            default_height=cfg.default_height,
            default_width=cfg.default_width,
        )
        for cfg in MODEL_REGISTRY.values()
        if cfg.kind == "image"
    ]


def _select_model(cfg_key: Optional[str], prompt: str) -> LocalModelConfig:
    if cfg_key:
        if cfg_key not in MODEL_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Model '{cfg_key}' not found")
        return MODEL_REGISTRY[cfg_key]
    return auto_select_image_model(prompt)


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    cfg = _select_model(req.model_key, req.prompt)
    final_prompt, final_negative = _build_prompts(
        req.prompt,
        req.negative_prompt,
        apply_quality_boost=req.apply_quality_boost,
        apply_artifact_cleanup=req.apply_artifact_cleanup,
    )

    # Try cache
    cached_img = get_cached_image(
        prompt=final_prompt,
        negative_prompt=final_negative,
        model_key=cfg.key,
        height=req.height,
        width=req.width,
        steps=req.steps,
        guidance=req.guidance,
        seed=req.seed,
        sampler=req.sampler,
    )
    cached = cached_img is not None
    image = cached_img

    if image is None:
        image, _ = _MODEL_MANAGER.generate_image(
            prompt=final_prompt,
            cfg=cfg,
            negative_prompt=final_negative,
            height=req.height,
            width=req.width,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            seed=req.seed,
            sampler=req.sampler,
        )

        set_cached_image(
            image=image,
            prompt=final_prompt,
            negative_prompt=final_negative,
            model_key=cfg.key,
            height=req.height,
            width=req.width,
            steps=req.steps,
            guidance=req.guidance,
            seed=req.seed,
            sampler=req.sampler,
        )

        log_generation(
            prompt=final_prompt,
            negative_prompt=final_negative,
            model_key=cfg.key,
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance=req.guidance,
            seed=req.seed,
            image=image,
        )

    img_b64 = _encode_image_to_base64(image)
    return GenerateResponse(
        model_key=cfg.key,
        prompt=final_prompt,
        negative_prompt=final_negative,
        height=req.height,
        width=req.width,
        steps=req.steps,
        guidance=req.guidance,
        seed=req.seed,
        sampler=req.sampler,
        cached=cached,
        image_base64=img_b64,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8008, reload=False)
