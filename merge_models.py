"""
Utility to merge 2–3 .safetensors models from M:\models into a new checkpoint,
and (optionally) auto-register it in model_registry.py for the Streamlit app.

Usage (in your venv):
    python merge_models.py

Requirements:
    pip install safetensors
"""

import os
from typing import List, Tuple
from safetensors.torch import load_file, save_file

# Adjust if needed
MODELS_DIR = r"M:\models"
MODEL_REGISTRY_PATH = "model_registry.py"  # assumed to be in the same folder as this script


def list_safetensors() -> List[str]:
    files = [
        f for f in os.listdir(MODELS_DIR)
        if f.lower().endswith(".safetensors")
    ]
    files.sort()
    return files


def choose_models(files: List[str]) -> List[int]:
    print("\nAvailable .safetensors in", MODELS_DIR)
    for i, name in enumerate(files):
        print(f"[{i}] {name}")
    print()

    while True:
        raw = input("Enter 2–3 model indices to merge (comma-separated, e.g. 0,2 or 1,3,4): ").strip()
        try:
            idxs = [int(x) for x in raw.split(",") if x.strip() != ""]
        except ValueError:
            print("❌ Invalid input, please enter indices like: 0,2 or 1,3,4")
            continue

        if not (2 <= len(idxs) <= 3):
            print("❌ Please select 2 or 3 models.")
            continue

        if any(i < 0 or i >= len(files) for i in idxs):
            print("❌ One or more indices are out of range.")
            continue

        return idxs


def get_weights(n: int) -> List[float]:
    print("\nNow enter weights for each selected model.")
    print("They will be normalized automatically so they sum to 1.0.")
    print("Example for 2 models: 0.7 and 0.3")

    weights = []
    for i in range(n):
        while True:
            raw = input(f"Weight for model #{i + 1}: ").strip()
            try:
                w = float(raw)
                if w <= 0:
                    print("Weight must be positive.")
                    continue
                weights.append(w)
                break
            except ValueError:
                print("❌ Please enter a numeric value, e.g. 0.7")

    s = sum(weights)
    normed = [w / s for w in weights]
    print("Normalized weights:", normed)
    return normed


def merge_states(
    model_paths: List[str],
    weights: List[float],
    output_path: str,
) -> None:
    """
    Merge safetensors from model_paths using given weights.
    We take the key set and shapes from the FIRST model as reference.
    """
    print("\n📦 Loading models...")
    states = [load_file(p) for p in model_paths]
    ref_state = states[0]

    print("🔗 Merging...")
    merged = {}
    for k, ref_tensor in ref_state.items():
        merged_tensor = ref_tensor * weights[0]
        for i in range(1, len(states)):
            other_state = states[i]
            if k in other_state and other_state[k].shape == ref_tensor.shape:
                merged_tensor = merged_tensor + other_state[k] * weights[i]
            else:
                # if missing or different shape, just keep contribution from ref model
                pass
        merged[k] = merged_tensor

    print("💾 Saving merged model ->", output_path)
    save_file(merged, output_path)
    print("✅ Done.")


def make_merged_filename(selected_files: List[str], weights: List[float]) -> str:
    # e.g. merge_morereal_0.7__rv_0.3.safetensors (shortened)
    parts = []
    for name, w in zip(selected_files, weights):
        stem = os.path.splitext(name)[0]
        short = stem[:10].replace(" ", "").replace(".", "_")
        parts.append(f"{short}_{w:.2f}")
    filename = "merged_" + "__".join(parts) + ".safetensors"
    # sanitize a bit
    filename = filename.replace("..", ".").replace("__", "_")
    return filename


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        ans = input(f"{prompt} {suffix} ").strip().lower()
        if ans == "" and default:
            return True
        if ans == "" and not default:
            return False
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y or n.")


def sanitize_key_from_filename(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    key = stem.lower()
    for ch in " -.:/\\()[]{}":
        key = key.replace(ch, "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key


def append_to_model_registry(merged_filename: str) -> None:
    if not os.path.exists(MODEL_REGISTRY_PATH):
        print(f"\n⚠️ {MODEL_REGISTRY_PATH} not found in current directory; "
              "cannot auto-register merged model.")
        return

    key = sanitize_key_from_filename(merged_filename)
    arch = input("\nEnter arch for this merged model (e.g. sdxl, flux) [sdxl]: ").strip() or "sdxl"

    snippet = f"""

# Auto-added by merge_models.py
_add(
    "{key}",
    "{merged_filename}",
    kind="image",
    arch="{arch}",
    tags=["merged", "custom"],
)
"""

    print(f"\n🧩 Appending this block to {MODEL_REGISTRY_PATH}:\n{snippet}")

    with open(MODEL_REGISTRY_PATH, "a", encoding="utf-8") as f:
        f.write(snippet)

    print(f"✅ Merged model registered as key '{key}'.")
    print("👉 Restart your Streamlit app so it picks up the new model.")


def main():
    print("=== Model Merger Utility ===")

    files = list_safetensors()
    if not files:
        print(f"No .safetensors files found in {MODELS_DIR}")
        return

    selected_idxs = choose_models(files)
    selected_files = [files[i] for i in selected_idxs]
    print("\nYou selected:")
    for i, name in enumerate(selected_files):
        print(f"  [{i}] {name}")

    weights = get_weights(len(selected_files))

    merged_name = make_merged_filename(selected_files, weights)
    merged_path = os.path.join(MODELS_DIR, merged_name)

    print("\nFinal merged file will be:")
    print("  ", merged_path)

    if not ask_yes_no("Proceed with merging?", default=True):
        print("Aborted.")
        return

    model_paths = [os.path.join(MODELS_DIR, name) for name in selected_files]
    merge_states(model_paths, weights, merged_path)

    if ask_yes_no("Add this merged model to model_registry.py so it appears in Streamlit UI?", default=True):
        append_to_model_registry(os.path.basename(merged_path))
    else:
        print("You can manually add it to model_registry.py later.")


if __name__ == "__main__":
    main()
