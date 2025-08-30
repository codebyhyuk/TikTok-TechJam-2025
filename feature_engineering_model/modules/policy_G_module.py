from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from detoxify import Detoxify
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

_MODEL: Optional[Detoxify] = None

def get_detoxify_model(model_name: str = "unbiased") -> Detoxify:
    global _MODEL
    if _MODEL is None:
        _MODEL = Detoxify(model_name)
    return _MODEL

def init_worker(model_name_for_worker: str):
    # Preload model once per process
    get_detoxify_model(model_name_for_worker)

def score_toxicity(text: str, model_name: str = "unbiased") -> Dict[str, float]:
    if not text or not str(text).strip():
        return {
            "toxicity": 0.0, "severe_toxicity": 0.0, "obscene": 0.0, "threat": 0.0,
            "insult": 0.0, "identity_attack": 0.0, "P_max": 0.0, "S_toxicity": 0.0
        }
    model = get_detoxify_model(model_name)
    preds = model.predict(str(text))
    probs = {
        "toxicity": float(preds.get("toxicity", 0.0)),
        "severe_toxicity": float(preds.get("severe_toxicity", preds.get("severe_toxic", 0.0))),
        "obscene": float(preds.get("obscene", 0.0)),
        "threat": float(preds.get("threat", 0.0)),
        "insult": float(preds.get("insult", 0.0)),
        "identity_attack": float(preds.get("identity_attack", preds.get("identity_hate", 0.0))),
    }
    P_max = max(probs.values())
    probs["P_max"] = float(P_max)
    probs["S_toxicity"] = float(1.0 - P_max)
    return probs

# --- TOP-LEVEL worker so it is picklable ---
def _policy_g_worker(idx_text_model: Tuple[int, str, str]) -> Tuple[int, Dict[str, float]]:
    idx, text, model_name = idx_text_model
    try:
        res = score_toxicity(text, model_name=model_name)
    except Exception:
        res = {
            "toxicity": 0.0, "severe_toxicity": 0.0, "obscene": 0.0, "threat": 0.0,
            "insult": 0.0, "identity_attack": 0.0, "P_max": 0.0, "S_toxicity": 0.0
        }
    return idx, res

def compute_policy_g_details_processed(
    df: pd.DataFrame,
    text_col: str = "text",
    model_name: str = "unbiased",
    max_workers: int = 8,
) -> pd.DataFrame:
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in DataFrame.")
    texts: List[str] = df[text_col].tolist()
    n = len(texts)

    cols = ["toxicity","severe_toxicity","obscene","threat","insult","identity_attack","P_max","S_toxicity"]
    out = {c: np.zeros(n, dtype=float) for c in cols}

    # Use initializer to load one model per process (avoid parent warm-up)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(model_name,)
    ) as ex:
        futures = [ex.submit(_policy_g_worker, (i, t, model_name)) for i, t in enumerate(texts)]
        for fut in as_completed(futures):
            idx, res = fut.result()
            for c in cols:
                out[c][idx] = float(res.get(c, 0.0))

    return pd.DataFrame(out, index=df.index)

def compute_policy_g_series_processed(
    df: pd.DataFrame,
    text_col: str = "text",
    model_name: str = "unbiased",
    max_workers: int = 2,
    field: str = "S_toxicity",  # keep default consistent with docstring
) -> pd.Series:
    details = compute_policy_g_details_processed(
        df, text_col=text_col, model_name=model_name, max_workers=max_workers
    )
    if field not in details.columns:
        raise KeyError(f"Field '{field}' not produced by score_toxicity().")
    return details[field].rename(f"policy_G_{field}")

if __name__ == "__main__":
    # Example usage guarded for multiprocessing
    df = pd.read_csv('/Users/evan/Documents/Projects/TikTok-TechJam-2025/data_gpt_labeler/final_data_2.csv')
    sample_df = df.sample(n=10, random_state=42).reset_index(drop=True).copy()

    print(sample_df["text"])
    print(sample_df["text"].iloc[9])

    policy_g_score = compute_policy_g_series_processed(sample_df)
    print(policy_g_score)