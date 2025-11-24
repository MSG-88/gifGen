# gif_generator.py

from typing import List
from io import BytesIO

import torch
from PIL import Image
import imageio


def generate_frames_with_pipeline(
    pipe,
    prompt: str,
    negative_prompt: str | None = None,
    num_frames: int = 8,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    seed: int | None = None,
) -> List[Image.Image]:
    """
    Basic implementation: generate multiple independent frames with varying seeds.

    For more cinematic motion, you’d replace this with:
      - latent interpolation, or
      - video diffusion pipeline, etc.
    """
    if seed is None:
        generator = None
    else:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    frames: List[Image.Image] = []
    for i in range(num_frames):
        # Slightly change seed per frame for variation
        frame_seed = None if generator is None else seed + i
        frame_generator = (
            None
            if frame_seed is None
            else torch.Generator(device=pipe.device).manual_seed(frame_seed)
        )

        with torch.autocast(pipe.device.type):
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=frame_generator,
            )

        image = out.images[0]
        frames.append(image)

    return frames


def frames_to_gif_bytes(
    frames: List[Image.Image],
    fps: int = 8,
    loop: int = 0,
) -> bytes:
    """
    Convert a list of PIL images to an in-memory GIF (as bytes).
    """
    if not frames:
        raise ValueError("No frames to convert to GIF")

    duration_ms = int(1000 / max(fps, 1))

    buf = BytesIO()
    imageio.mimsave(
        buf,
        frames,
        format="GIF",
        duration=duration_ms / 1000.0,  # seconds per frame
        loop=loop,
    )
    buf.seek(0)
    return buf.getvalue()
