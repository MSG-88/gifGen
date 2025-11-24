# config.py

from pathlib import Path

# Root directory where your local diffusers models live
MODELS_ROOT = Path(r"M:\models")

# Map a user-friendly name to a local path
LOCAL_MODELS = {
    "SDXL (local)": MODELS_ROOT / "sdxl",
    # "MyCustomModel": MODELS_ROOT / "my_custom_model",
}

# Default image settings
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
MAX_WIDTH = 1536
MAX_HEIGHT = 1536

# GIF defaults
DEFAULT_NUM_FRAMES = 8
DEFAULT_FPS = 8
