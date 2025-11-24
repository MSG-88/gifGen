# model_manager.py

import os
from typing import Dict, Optional

import torch
import diffusers
from diffusers import DiffusionPipeline
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)

from model_registry import LocalModelConfig
from router import auto_select_image_model


def _pick_single_file_loader():
    """
    Try to find a usable `.from_single_file` method.
    """
    # 1) Generic DiffusionPipeline
    if hasattr(DiffusionPipeline, "from_single_file"):
        return DiffusionPipeline.from_single_file

    # 2) Try SD / SDXL-specific
    StableDiffusionPipeline = None
    StableDiffusionXLPipeline = None
    try:
        from diffusers import StableDiffusionPipeline as SDP  # type: ignore
        StableDiffusionPipeline = SDP
    except Exception:
        pass

    try:
        from diffusers import StableDiffusionXLPipeline as SDXLP  # type: ignore
        StableDiffusionXLPipeline = SDXLP
    except Exception:
        pass

    if StableDiffusionXLPipeline is not None and hasattr(StableDiffusionXLPipeline, "from_single_file"):
        return StableDiffusionXLPipeline.from_single_file

    if StableDiffusionPipeline is not None and hasattr(StableDiffusionPipeline, "from_single_file"):
        return StableDiffusionPipeline.from_single_file

    return None


_SINGLE_FILE_LOADER = _pick_single_file_loader()
_DEFAULT_SAMPLER = "dpmpp_2m_karras"


def _make_scheduler(pipe: DiffusionPipeline, sampler: str):
    # Prefer Karras for sharper detail when available.
    if not sampler or sampler == "default":
        return pipe.scheduler

    config = pipe.scheduler.config
    if sampler == "dpmpp_2m_karras":
        return DPMSolverMultistepScheduler.from_config(
            config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
    if sampler == "euler_a":
        return EulerAncestralDiscreteScheduler.from_config(config)
    return pipe.scheduler


def _configure_pipe(pipe: DiffusionPipeline, device: str, sampler: str) -> DiffusionPipeline:
    try:
        pipe.scheduler = _make_scheduler(pipe, sampler)
    except Exception:
        # fallback silently to the pipeline's default scheduler
        pass

    pipe.to(device)

    if device == "cuda":
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
    if hasattr(pipe, "enable_attention_slicing"):
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    if hasattr(pipe, "enable_vae_tiling"):
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass

    pipe.set_progress_bar_config(disable=True)
    return pipe


class ModelManager:
    def __init__(self, device: Optional[str] = None, dtype: torch.dtype = torch.float16):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype
        self._pipelines: Dict[str, DiffusionPipeline] = {}

    def _load_image_pipeline(self, cfg: LocalModelConfig, sampler: str) -> DiffusionPipeline:
        cache_key = f"{cfg.key}:{sampler or 'default'}"
        if cache_key in self._pipelines:
            return self._pipelines[cache_key]

        if _SINGLE_FILE_LOADER is None:
            raise RuntimeError(
                "Your diffusers installation does not expose any `from_single_file` "
                "method on DiffusionPipeline / SD / SDXL classes.\n"
                "Fix by upgrading in this env:\n"
                "  python -m pip install -U 'diffusers>=0.28.0'\n"
            )

        if not os.path.exists(cfg.path):
            raise FileNotFoundError(f"Model file not found: {cfg.path}")

        pipe = _SINGLE_FILE_LOADER(
            cfg.path,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )

        pipe = _configure_pipe(pipe, self.device, sampler or "default")
        self._pipelines[cache_key] = pipe
        return pipe

    def generate_image(
        self,
        prompt: str,
        cfg: Optional[LocalModelConfig] = None,
        negative_prompt: str = "",
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 5.5,
        seed: Optional[int] = None,
        sampler: str = _DEFAULT_SAMPLER,
    ):
        if cfg is None:
            cfg = auto_select_image_model(prompt)

        pipe = self._load_image_pipeline(cfg, sampler)

        if height is None:
            height = cfg.default_height
        if width is None:
            width = cfg.default_width

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        image = result.images[0]
        return image, cfg
