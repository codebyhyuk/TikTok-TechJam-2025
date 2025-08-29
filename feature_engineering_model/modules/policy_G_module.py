from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from detoxify import Detoxify
from concurrent.futures import ThreadPoolExecutor, as_completed

_MODEL: Optional[Detoxify] = None

def get_detoxify_model(model_name: str= "unbiased") -> Detoxify:
    # returns a cached Detoxify model instance
    global _MODEL
    if _MODEL is None:
        _MODEL = Detoxify(model_name)
    return _MODEL

_TOX_KEYS_PRIMARY = (
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
)

_TOX_KEYS_FALLBACKS = {
    "severe_toxicity": ("severe_toxic",),
    "identity_attack": ("identity_hate",),
}
 

def score_toxicity(text: str, model_name: str = "unbiased") -> Dict[str, float]:
    '''
    Returns:
    {
      "toxicity": ...,
      "severe_toxicity": ...,
      "obscene": ...,
      "threat": ...,
      "insult": ...,
      "identity_attack": ...,
      "P_max": ...,         # worst-case toxicity (max of the 6 labels)
      "S_toxicity": ...     # 1 - P_max
    }
    '''
    if not text or not str(text).strip():
        return {
            "toxicity": 0.0,
            "severe_toxicity": 0.0,
            "obscene": 0.0,
            "threat": 0.0,
            "insult": 0.0,
            "identity_attack": 0.0,
            "P_max": 0.0,
            "S_toxicity": 0.0
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
    S_toxicity = 1.0 - P_max

    probs["P_max"] = float(P_max)
    probs["S_toxicity"] = float(S_toxicity)
    return probs


def compute_policy_g_details_threaded(
    df: pd.DataFrame,
    text_col: str = "text",
    model_name: str = "unbiased",
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Threaded computation of Policy G details for each row, preserving the original
    score_toxicity() output fields.

    Returns:
        pd.DataFrame with columns:
        ["toxicity","severe_toxicity","obscene","threat","insult","identity_attack","P_max","S_toxicity"]
        aligned to df.index.
    """
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in DataFrame.")

    # warm up a shared model once (shared across threads)
    _ = get_detoxify_model(model_name)

    texts: List[str] = df[text_col].tolist()
    n = len(texts)

    # preallocate containers
    cols = ["toxicity","severe_toxicity","obscene","threat","insult","identity_attack","P_max","S_toxicity"]
    out = {c: np.zeros(n, dtype=float) for c in cols}

    def worker(idx: int, text: str) -> Tuple[int, Dict[str, float]]:
        try:
            res = score_toxicity(text, model_name=model_name)
        except Exception:
            res = {c: 0.0 for c in cols}
        return idx, res

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, t in enumerate(texts):
            futures.append(ex.submit(worker, i, t))

        for fut in as_completed(futures):
            idx, res = fut.result()
            for c in cols:
                out[c][idx] = float(res.get(c, 0.0))

    df_out = pd.DataFrame(out, index=df.index)
    return df_out


def compute_policy_g_series_threaded(
    df: pd.DataFrame,
    text_col: str = "text",
    model_name: str = "unbiased",
    max_workers: int = 8,
    field: str = "S_toxicity",
) -> pd.Series:
    """
    Convenience wrapper that returns a single Series (0-1) for a chosen field
    from the original score_toxicity() output. Default is 'P_max'.

    Example fields: 'P_max', 'S_toxicity', 'toxicity', 'insult', etc.
    """
    details = compute_policy_g_details_threaded(
        df,
        text_col=text_col,
        model_name=model_name,
        max_workers=max_workers,
    )
    if field not in details.columns:
        raise KeyError(f"Field '{field}' not produced by score_toxicity().")
    return details[field].rename(f"policy_G_{field}")

# --------------------------------------------------------------------------------------------------------------------------
df = pd.read_csv('/Users/evan/Documents/Projects/TikTok-TechJam-2025/data_gpt_labeler/final_data_2.csv')

sample_df = df.sample(n=10, random_state=42).reset_index(drop=True).copy()

print(sample_df["text"])
print(sample_df["text"].iloc[9])

policy_g_score = compute_policy_g_series_threaded(sample_df)

print(policy_g_score)


