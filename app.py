# app.py

import os
from typing import Optional

import streamlit as st
from PIL import Image

from model_registry import MODEL_REGISTRY, LocalModelConfig
from model_manager import ModelManager
from router import auto_select_image_model
from cache_backend import get_cached_image, set_cached_image
from vector_store import init_schema, log_generation, search_by_prompt


QUALITY_BOOST = "ultra-detailed, sharp focus, high dynamic range, 8k, photorealistic textures, crisp edges"
NEGATIVE_CLEANUP = "lowres, watermark, text, logo, oversaturated, deformed, extra limbs, artifacts, noise"
SAMPLER_CHOICES = {
    "dpmpp_2m_karras": "DPM++ 2M Karras (sharper, stable)",
    "euler_a": "Euler a (fast)",
    "default": "Model default",
}


@st.cache_resource
def get_model_manager() -> ModelManager:
    return ModelManager()


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


def select_model_ui() -> Optional[LocalModelConfig]:
    options = ["Auto"] + [
        cfg.key for cfg in MODEL_REGISTRY.values() if cfg.kind == "image"
    ]
    choice = st.selectbox("Model (Auto = let app choose):", options)
    if choice == "Auto":
        return None
    return MODEL_REGISTRY[choice]


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

        chosen_cfg = select_model_ui()

        default_h = 1024
        default_w = 1024
        if chosen_cfg is not None and chosen_cfg.default_height > 0:
            default_h = chosen_cfg.default_height
            default_w = chosen_cfg.default_width

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

        mgr = get_model_manager()

        if chosen_cfg is None:
            auto_cfg = auto_select_image_model(prompt)
            auto_info_placeholder.info(f"Auto-selected model: `{auto_cfg.key}`")
        else:
            auto_cfg = chosen_cfg
            auto_info_placeholder.info(f"Using model: `{auto_cfg.key}`")

        if not use_seed:
            seed_value = None
        else:
            seed_value = int(seed)

        final_prompt, final_negative = build_prompts(
            prompt,
            negative_prompt,
            apply_quality_boost=apply_quality_boost,
            apply_artifact_cleanup=apply_artifact_cleanup,
        )

        # 1) Try Redis cache first
        cached = get_cached_image(
            prompt=final_prompt,
            negative_prompt=final_negative,
            model_key=auto_cfg.key,
            height=height,
            width=width,
            steps=steps,
            guidance=guidance,
            seed=seed_value,
            sampler=sampler,
        )

        if cached is not None:
            st.success("Loaded from Redis cache")
            image = cached
            used_cfg = auto_cfg
        else:
            with st.spinner("Generating image..."):
                image, used_cfg = mgr.generate_image(
                    prompt=final_prompt,
                    cfg=auto_cfg,
                    negative_prompt=final_negative,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    seed=seed_value,
                    sampler=sampler,
                )

            # Store in Redis
            set_cached_image(
                image=image,
                prompt=final_prompt,
                negative_prompt=final_negative,
                model_key=used_cfg.key,
                height=height,
                width=width,
                steps=steps,
                guidance=guidance,
                seed=seed_value,
                sampler=sampler,
            )

            # Log in DuckDB
            log_generation(
                prompt=final_prompt,
                negative_prompt=final_negative,
                model_key=used_cfg.key,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                seed=seed_value,
                image=image,
            )

        st.subheader("Result")
        st.image(image, use_column_width=True, caption=f"Model: {used_cfg.key}")

        st.download_button(
            "Download PNG",
            data=image_to_bytes(image),
            file_name="generated.png",
            mime="image/png",
        )


if __name__ == "__main__":
    main()
