# vector_store.py

import os
import datetime
from functools import lru_cache
from typing import Optional, List, Tuple

import duckdb
import numpy as np
from PIL import Image

import torch
from transformers import CLIPModel, CLIPProcessor

DB_PATH = "data/generations.duckdb"
IMAGE_DIR = "data/images"

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# --- CLIP singleton ---

_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_CLIP_DEVICE = None


def _get_clip():
    global _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE
    if _CLIP_MODEL is None:
        model_name = "openai/clip-vit-large-patch14"
        _CLIP_MODEL = CLIPModel.from_pretrained(model_name)
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name)
        _CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _CLIP_MODEL.to(_CLIP_DEVICE)
    return _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE


def _connect():
    return duckdb.connect(DB_PATH)


def init_schema():
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS generations (
            -- DuckDB versions bundled with Streamlit often lack primary key/identity support;
            -- keep a manual BIGINT id instead.
            id BIGINT,
            prompt TEXT,
            negative_prompt TEXT,
            model_key TEXT,
            width INT,
            height INT,
            steps INT,
            guidance DOUBLE,
            seed BIGINT,
            created_at TIMESTAMP,
            image_path TEXT,
            text_emb BLOB
        );
        """
    )
    conn.close()


@lru_cache(maxsize=256)
def _cached_text_emb_bytes(prompt: str) -> bytes:
    """
    Cache CLIP text embeddings per prompt to avoid recomputing for repeats.
    Returns normalized float32 embedding as raw bytes.
    """
    model, processor, device = _get_clip()
    inputs = processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = model.get_text_features(**inputs)  # (1, 768)
    emb = feats[0].cpu().float().numpy()
    norm = np.linalg.norm(emb) + 1e-12
    emb = emb / norm
    return emb.astype("float32").tobytes()


@torch.no_grad()
def compute_text_embedding(prompt: str) -> np.ndarray:
    # Copy to avoid callers mutating the cached buffer
    return np.frombuffer(_cached_text_emb_bytes(prompt), dtype="float32").copy()


def log_generation(
    prompt: str,
    negative_prompt: str,
    model_key: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: Optional[int],
    image: Image.Image,
):
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{ts}_{model_key}.png"
    image_path = os.path.join(IMAGE_DIR, filename)
    image.save(image_path, format="PNG")

    emb = compute_text_embedding(prompt)
    emb_bytes = emb.tobytes()

    conn = _connect()
    next_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM generations").fetchone()[0]
    conn.execute(
        """
        INSERT INTO generations (
            id,
            prompt, negative_prompt, model_key,
            width, height, steps, guidance, seed,
            created_at, image_path, text_emb
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            next_id,
            prompt,
            negative_prompt,
            model_key,
            width,
            height,
            steps,
            guidance,
            None if seed is None else int(seed),
            datetime.datetime.utcnow(),
            image_path,
            emb_bytes,
        ],
    )
    conn.close()


def search_by_prompt(query: str, top_k: int = 8) -> List[Tuple]:
    """
    Return top_k similar generations to the query prompt.
    Each row: (id, prompt, model_key, image_path, similarity)
    """
    q_emb = compute_text_embedding(query)

    conn = _connect()
    rows = conn.execute(
        """
        SELECT id, prompt, model_key, image_path, text_emb
        FROM generations
        """
    ).fetchall()
    conn.close()

    meta: List[Tuple[int, str, str, str]] = []
    emb_list: List[np.ndarray] = []
    for (id_val, p, mk, img_path, emb_bytes) in rows:
        emb = np.frombuffer(emb_bytes, dtype="float32")
        if emb.shape != q_emb.shape:
            continue
        meta.append((id_val, p, mk, img_path))
        emb_list.append(emb)

    if not emb_list:
        return []

    emb_matrix = np.stack(emb_list, axis=0)  # (n, d)
    sims = emb_matrix @ q_emb  # vectorized dot product

    order = np.argsort(-sims)[:top_k]
    results = []
    for idx in order:
        id_val, p, mk, img_path = meta[int(idx)]
        results.append((id_val, p, mk, img_path, float(sims[int(idx)])))
    return results
