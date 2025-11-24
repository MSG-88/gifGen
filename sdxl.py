import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from tqdm import tqdm


# ============ Model Manager ============

@dataclass
class ModelConfig:
    model_path: str
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    lora_path: Optional[str] = None
    lora_weight: float = 0.7


class ModelManager:
    """
    Loads and configures an SDXL pipeline from a local .safetensors file.
    Optionally applies a single LoRA.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._pipe: Optional[StableDiffusionXLPipeline] = None

    def load(self) -> StableDiffusionXLPipeline:
        if self._pipe is not None:
            return self._pipe

        print(f"[ModelManager] Loading SDXL from: {self.config.model_path}")
        pipe = StableDiffusionXLPipeline.from_single_file(
            self.config.model_path,
            torch_dtype=self.config.dtype,
        )

        # Performance / VRAM tweaks (good for 16GB)
        pipe.to(self.config.device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            print("[ModelManager] xFormers not available, continuing without it.")

        pipe.enable_vae_tiling()
        pipe.enable_vae_slicing()

        # Optional: offload some stuff to CPU if you run tight on VRAM
        # requires accelerate
        # from accelerate import cpu_offload
        # pipe.enable_model_cpu_offload()

        if self.config.lora_path:
            print(f"[ModelManager] Loading LoRA: {self.config.lora_path}")
            pipe.load_lora_weights(self.config.lora_path)
            pipe.fuse_lora(lora_scale=self.config.lora_weight)

        self._pipe = pipe
        return pipe

    @property
    def pipe(self) -> StableDiffusionXLPipeline:
        if self._pipe is None:
            return self.load()
        return self._pipe


# ============ Image Generator ============

@dataclass
class ImageGenConfig:
    width: int = 896
    height: int = 1152
    steps: int = 25
    cfg_scale: float = 6.5
    seed: int = 222123
    negative_prompt: str = (
        "low-res, artifacts, deformed, disfigured, blurry, extra limbs, "
        "worst quality, jpeg artifacts, text, watermark"
    )


class ImageGenerator:
    """
    Uses a StableDiffusionXLPipeline to generate individual frames or sequences.
    """

    def __init__(self, pipe: StableDiffusionXLPipeline, config: ImageGenConfig):
        self.pipe = pipe
        self.config = config

    def _make_generator(self, seed_offset: int = 0) -> torch.Generator:
        g = torch.Generator(device=self.pipe.device)
        g.manual_seed(self.config.seed + seed_offset)
        return g

    def generate_frame(
        self,
        prompt: str,
        seed_offset: int = 0,
    ) -> Image.Image:
        """
        Generate a single PIL image.
        seed_offset lets you slightly vary the noise per frame if you want motion.
        """
        generator = self._make_generator(seed_offset)
        result = self.pipe(
            prompt=prompt,
            negative_prompt=self.config.negative_prompt,
            num_inference_steps=self.config.steps,
            guidance_scale=self.config.cfg_scale,
            width=self.config.width,
            height=self.config.height,
            generator=generator,
        )
        return result.images[0]

    def generate_sequence(
        self,
        prompts: List[str],
        out_dir: str,
        vary_seed: bool = True,
        overwrite: bool = True,
    ) -> List[str]:
        """
        Generate one frame per prompt and save as PNGs: frame_0000.png, frame_0001.png, ...

        Returns: list of filepaths.
        """
        os.makedirs(out_dir, exist_ok=True)
        frame_paths: List[str] = []

        if not overwrite and any(f.startswith("frame_") for f in os.listdir(out_dir)):
            raise RuntimeError(f"Output dir '{out_dir}' already has frames and overwrite=False")

        print(f"[ImageGenerator] Generating {len(prompts)} frames into: {out_dir}")

        for idx, prompt in enumerate(tqdm(prompts, desc="Generating frames")):
            seed_offset = idx if vary_seed else 0
            img = self.generate_frame(prompt=prompt, seed_offset=seed_offset)
            frame_path = os.path.join(out_dir, f"frame_{idx:04d}.png")
            img.save(frame_path)
            frame_paths.append(frame_path)

        print(f"[ImageGenerator] Done. {len(frame_paths)} frames saved.")
        return frame_paths


# ============ Video Generator ============

@dataclass
class VideoConfig:
    fps: int = 16
    codec: str = "libx264"
    crf: int = 18  # quality; 0-51, lower = better


class VideoGenerator:
    """
    Uses ffmpeg (CLI) to turn a folder of frames into a video.
    Assumes ffmpeg is installed and available on PATH.
    """

    def __init__(self, config: VideoConfig):
        self.config = config

    def frames_to_video(
        self,
        frames_dir: str,
        output_path: str,
        pattern: str = "frame_%04d.png",
    ) -> None:
        """
        Run ffmpeg over frames_dir/pattern -> output_path.
        """
        print(f"[VideoGenerator] Creating video from frames in: {frames_dir}")
        input_pattern = os.path.join(frames_dir, pattern)

        cmd = [
            "ffmpeg",
            "-y",  # overwrite
            "-framerate",
            str(self.config.fps),
            "-i",
            input_pattern,
            "-c:v",
            self.config.codec,
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(self.config.crf),
            output_path,
        ]

        print("[VideoGenerator] Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            print(
                "[VideoGenerator][ERROR] ffmpeg not found. "
                "Install ffmpeg and ensure it's on your PATH."
            )
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print("[VideoGenerator][ERROR] ffmpeg failed:", e)
            sys.exit(1)

        print(f"[VideoGenerator] Video written to: {output_path}")


# ============ Prompt utilities ============

def load_prompts(
    base_prompt: str,
    prompt_file: Optional[str],
    num_frames: int,
) -> List[str]:
    """
    If prompt_file is provided, read one prompt per line.
    Otherwise, repeat base_prompt num_frames times.
    """
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines()]
        prompts = [ln for ln in lines if ln]
        print(f"[Prompts] Loaded {len(prompts)} prompts from {prompt_file}")
        return prompts

    # Simple case: static prompt repeated
    return [base_prompt] * num_frames


# ============ CLI ============

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end AI video creator (SDXL -> frames -> MP4)"
    )

    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to SDXL .safetensors file",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Optional LoRA .safetensors path",
    )
    parser.add_argument(
        "--lora-weight",
        type=float,
        default=0.7,
        help="LoRA weight/scale (default 0.7)",
    )

    # Prompts / frames
    parser.add_argument(
        "--prompt",
        type=str,
        default="cinematic portrait, golden hour, 35mm lens, high detail, SDXL",
        help="Base prompt (ignored if --prompt-file is provided)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional text file: one prompt per line, used per frame",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=24,
        help="Number of frames (used only if --prompt-file not provided)",
    )

    # Image generation
    parser.add_argument("--width", type=int, default=896, help="Image width")
    parser.add_argument("--height", type=int, default=1152, help="Image height")
    parser.add_argument("--steps", type=int, default=25, help="Diffusion steps")
    parser.add_argument("--cfg-scale", type=float, default=6.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=222123, help="Base random seed")
    parser.add_argument(
        "--no-vary-seed",
        action="store_true",
        help="If set, use exact same seed for all frames (less motion, more consistency)",
    )

    # Output
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Directory to save frames and video",
    )
    parser.add_argument(
        "--frames-subdir",
        type=str,
        default="frames",
        help="Subdirectory under out-dir for frames",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default="output.mp4",
        help="Video filename (under out-dir)",
    )
    parser.add_argument("--fps", type=int, default=16, help="Video FPS")
    parser.add_argument("--crf", type=int, default=18, help="Video quality (0-51)")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, args.frames_subdir)
    video_path = os.path.join(args.out_dir, args.video_name)

    # 1) Load prompts
    prompts = load_prompts(args.prompt, args.prompt_file, args.num_frames)

    # 2) Model
    model_cfg = ModelConfig(
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
        lora_path=args.lora_path,
        lora_weight=args.lora_weight,
    )
    model_manager = ModelManager(model_cfg)
    pipe = model_manager.pipe

    # 3) Image generation
    img_cfg = ImageGenConfig(
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
    )
    image_gen = ImageGenerator(pipe, img_cfg)
    frame_paths = image_gen.generate_sequence(
        prompts=prompts,
        out_dir=frames_dir,
        vary_seed=not args.no_vary_seed,
        overwrite=True,
    )

    if not frame_paths:
        print("[main] No frames generated, aborting.")
        sys.exit(1)

    # 4) Video creation
    vid_cfg = VideoConfig(
        fps=args.fps,
        crf=args.crf,
        codec="libx264",
    )
    video_gen = VideoGenerator(vid_cfg)
    video_gen.frames_to_video(frames_dir=frames_dir, output_path=video_path)

    print("[main] All done.")


if __name__ == "__main__":
    main()
