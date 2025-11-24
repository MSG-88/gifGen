# app.py

import os
import re
import base64
from typing import Optional, List, Dict, Any

import requests
import streamlit as st
from PIL import Image

from model_registry import MODEL_REGISTRY
from vector_store import init_schema, search_by_prompt


QUALITY_BOOST = "ultra-detailed, sharp focus, high dynamic range, 8k, photorealistic textures, crisp edges"
NEGATIVE_CLEANUP = "lowres, watermark, text, logo, oversaturated, deformed, extra limbs, artifacts, noise"
SAMPLER_CHOICES = {
    "dpmpp_2m_karras": "DPM++ 2M Karras (sharper, stable)",
    "euler_a": "Euler a (fast)",
    "default": "Model default",
}
API_BASE_DEFAULT = os.getenv("GEN_API_BASE", "http://127.0.0.1:8000")


@st.cache_resource
def _init_vector_store_once():
    init_schema()
    return True


def image_to_bytes(image: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def _normalize_models(remote_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    for m in remote_models:
        if m.get("kind") != "image":
            continue
        models.append(
            {
                "key": m.get("key"),
                "name": m.get("name", m.get("key")),
                "arch": m.get("arch", ""),
                "tags": m.get("tags", []),
                "default_height": m.get("default_height", 1024),
                "default_width": m.get("default_width", 1024),
            }
        )
    return models


@st.cache_data(show_spinner=False, ttl=60)
def fetch_remote_models(api_base: str) -> List[Dict[str, Any]]:
    try:
        url = api_base.rstrip("/") + "/models"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def select_model_ui(remote_models: List[Dict[str, Any]]) -> tuple[Optional[str], int, int]:
    models = _normalize_models(remote_models)

    if not models:
        models = [
            {
                "key": cfg.key,
                "name": cfg.name,
                "arch": getattr(cfg, "arch", ""),
                "tags": getattr(cfg, "tags", []),
                "default_height": cfg.default_height,
                "default_width": cfg.default_width,
            }
            for cfg in MODEL_REGISTRY.values()
            if cfg.kind == "image"
        ]

    options = ["Auto"] + [m["key"] for m in models]
    choice = st.selectbox("Model (Auto = API auto-select):", options)

    default_h, default_w = 1024, 1024
    if choice == "Auto":
        if models:
            default_h = models[0]["default_height"]
            default_w = models[0]["default_width"]
        return None, default_h, default_w

    for m in models:
        if m["key"] == choice:
            default_h = m["default_height"]
            default_w = m["default_width"]
            break

    return choice, default_h, default_w


def build_prompts(prompt: str, negative_prompt: str, apply_quality_boost: bool, apply_artifact_cleanup: bool) -> tuple[str, str]:
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


def _call_api_generate(api_base: str, payload: dict) -> dict:
    url = api_base.rstrip("/") + "/generate"
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _decode_base64_image(b64_str: str) -> Image.Image:
    import io

    data = base64.b64decode(b64_str)
    buf = io.BytesIO(data)
    return Image.open(buf).convert("RGB")


def generate_story_pages(base_prompt: str, storyline: str, page_count: int) -> List[str]:
    """
    Deterministically expand a storyline into per-page prompts.
    Splits the storyline into sentences; if fewer than pages, pads with continuations.
    """
    base_prompt = base_prompt.strip()
    storyline = storyline.strip()

    scenes: List[str] = []
    if storyline:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", storyline) if s.strip()]
        scenes.extend(sentences)

    prompts: List[str] = []
    for idx in range(page_count):
        if idx < len(scenes):
            scene = scenes[idx]
        else:
            scene = f"{base_prompt} - continuation scene {idx + 1}"
        prompts.append(
            f"{scene}, magazine spread page {idx + 1}, consistent characters, coherent storyline, cinematic framing, sequential art layout"
        )
    return prompts


def main():
    st.set_page_config(
        page_title="Local Hi-Res Image Generator",
        layout="wide",
    )

    st.title("Local Hi-Res Image Generator")
    st.caption("Using local .safetensors models (SDXL / Flux) with Redis cache & DuckDB memory.")

    _init_vector_store_once()

    with st.sidebar:
        st.header("Generation Settings")
        api_base = st.text_input("API base URL", API_BASE_DEFAULT)
        remote_models = fetch_remote_models(api_base)
        if remote_models:
            st.caption(f"FastAPI models loaded ({len(remote_models)})")
        else:
            st.caption("FastAPI models unavailable, falling back to local registry")

        selected_model_key, default_h, default_w = select_model_ui(remote_models)

        height = st.slider("Height", min_value=512, max_value=1536, step=64, value=default_h)
        width = st.slider("Width", min_value=512, max_value=1536, step=64, value=default_w)

        steps = st.slider("Steps", min_value=10, max_value=60, step=2, value=28)
        guidance = st.slider("CFG (guidance)", min_value=1.0, max_value=12.0, step=0.5, value=5.5)
        seed = st.number_input("Seed (optional)", min_value=0, max_value=2**31 - 1, value=0)
        use_seed = st.checkbox("Lock seed", value=False)

        negative_prompt = st.text_area(
            "Negative prompt (optional)",
            value="low quality, blurry, bad anatomy, extra limbs",
            height=80,
        )
        apply_quality_boost = st.checkbox("Quality boost (clarity, sharpness)", value=True)
        apply_artifact_cleanup = st.checkbox("Artifact cleanup", value=True)
        sampler_labels = list(SAMPLER_CHOICES.values())
        sampler_keys = list(SAMPLER_CHOICES.keys())
        sampler_idx = 0
        sampler_choice = st.selectbox("Sampler", sampler_labels, index=sampler_idx)
        sampler = sampler_keys[sampler_labels.index(sampler_choice)]

    col_main, col_side = st.columns([2.5, 1.5])

    with col_main:
        st.subheader("Prompt")
        prompt = st.text_area(
            "Describe the image:",
            height=150,
            placeholder="e.g. ultra-realistic portrait of a woman in cyberpunk city at night, neon lights, 85mm lens",
        )
        multi_page = st.checkbox("Comic / magazine multi-page", value=False)
        page_count = 4
        if multi_page:
            page_count = st.slider("Number of pages", min_value=2, max_value=12, value=4, step=1)
            storyline_text = st.text_area(
                "Storyline (auto-split into pages)",
                height=120,
                placeholder="Enter the storyline; sentences will map to pages. Remaining pages will continue the main prompt.",
            )
        else:
            storyline_text = ""

        generate_btn = st.button("Generate", type="primary")
        auto_info_placeholder = st.empty()

    with col_side:
        st.subheader("Search previous results")
        ref_query = st.text_input("Find similar past prompts:")
        if ref_query.strip():
            results = search_by_prompt(ref_query.strip(), top_k=4)
            for (id_val, p, mk, img_path, sim) in results:
                st.markdown(f"**Prompt:** {p}")
                st.caption(f"Model: `{mk}` | Similarity: {sim:.3f}")
                if os.path.exists(img_path):
                    st.image(img_path, width=200)
                st.markdown("---")

    if generate_btn:
        if not prompt.strip():
            st.warning("Please enter a prompt first.")
            return

        if selected_model_key is None:
            auto_info_placeholder.info("Using API auto-selected model")
        else:
            auto_info_placeholder.info(f"Using model: `{selected_model_key}`")

        seed_value = int(seed) if use_seed else None
        base_prompt = prompt.strip()
        base_negative = negative_prompt

        def _generate_via_api(p_txt: str, n_txt: str, page_idx: int | None = None):
            payload = {
                "prompt": p_txt,
                "negative_prompt": n_txt,
                "model_key": selected_model_key,
                "height": height,
                "width": width,
                "steps": steps,
                "guidance": guidance,
                "seed": seed_value,
                "sampler": sampler,
                "apply_quality_boost": apply_quality_boost,
                "apply_artifact_cleanup": apply_artifact_cleanup,
            }
            desc = f"Generating page {page_idx}/{page_count}..." if page_idx else "Generating image..."
            with st.spinner(desc):
                try:
                    resp = _call_api_generate(api_base, payload)
                except Exception as e:
                    st.error(f"API generation failed{' (page ' + str(page_idx) + ')' if page_idx else ''}: {e}")
                    return None, None

            try:
                img = _decode_base64_image(resp.get("image_base64", ""))
            except Exception as e:
                st.error(f"Could not decode image{' (page ' + str(page_idx) + ')' if page_idx else ''}: {e}")
                return None, None

            if resp.get("cached"):
                st.success(f"Loaded from API cache{' (page ' + str(page_idx) + ')' if page_idx else ''}")
            used_model = resp.get("model_key", selected_model_key or "auto")
            return img, used_model

        if multi_page:
            st.subheader("Result (multi-page)")
            images = []
            used_model_key = selected_model_key or "auto"
            page_prompts = generate_story_pages(base_prompt, storyline_text, page_count)

            with st.expander("Planned pages", expanded=False):
                for idx, p_txt in enumerate(page_prompts, start=1):
                    st.markdown(f"**Page {idx}:** {p_txt}")

            for idx, page_prompt in enumerate(page_prompts, start=1):
                img, used_model_key = _generate_via_api(page_prompt, base_negative, page_idx=idx)
                if img is None:
                    return
                images.append((idx, img, used_model_key))

            cols = st.columns(2)
            for i, (idx, img, used_model_key) in enumerate(images):
                with cols[i % 2]:
                    st.image(img, use_column_width=True, caption=f"Page {idx} - Model: {used_model_key}")
                    st.download_button(
                        f"Download Page {idx}",
                        data=image_to_bytes(img),
                        file_name=f"page_{idx}.png",
                        mime="image/png",
                        key=f"dl_page_{idx}",
                    )
        else:
            image, used_model_key = _generate_via_api(base_prompt, base_negative)
            if image is None:
                return
            st.subheader("Result")
            st.image(image, use_column_width=True, caption=f"Model: {used_model_key}")

            st.download_button(
                "Download PNG",
                data=image_to_bytes(image),
                file_name="generated.png",
                mime="image/png",
            )


if __name__ == "__main__":
    main()
