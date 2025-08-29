from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

INTENT_LABELS = ['genuine', 'spam', 'advertising', 'competitor attack', "incentivize", "mistaken identity"]

ZSC_MODEL_NAME = "facebook/bart-large-mnli"

_ZSC_PIPELINE = None

def get_zero_shot_pipeline(model_name: str = ZSC_MODEL_NAME):
    global _ZSC_PIPELINE
    # check if is the initial state == None
    if _ZSC_PIPELINE is None:
        _ZSC_PIPELINE = pipeline(
            task = "zero-shot-classification",
            model = model_name
        )
    return _ZSC_PIPELINE


def score_intent(
    text:str,
    labels: Optional[List[str]] = None,
    model_name: str = ZSC_MODEL_NAME,
) -> Dict[str, float]:
    # returns a dict: {label: probability}, including S(intent) = P("genuine").
    
    if not text or not text.strip():
        # edge case
        # if empty text, return 0 probability for all labels, including intent
        base = {lbl: 0.0 for lbl in labels or INTENT_LABELS}
        base["S_intent"] = 0.0
        return base
    
    zsc = get_zero_shot_pipeline(model_name)
    use_labels = labels or INTENT_LABELS
    
    res = zsc(
        sequences=text,
        candidate_labels = use_labels,
        multi_label = False # pick one distribution that sums ~1
    )
    
    # mapping scores to their labels
    scores = dict(zip(res["labels"], res['scores']))
    
    # ensuring all labels exist, incase the model drops something
    for lbl in use_labels:
        scores.setdefault(lbl, 0.0)
    
    # S(intent) = P("genuine")
    scores["S_intent"] = float(scores.get("genuine", 0.0))
    return scores


def compute_intent_score_for_row(row):
    return score_intent(row['text'])


def compute_intent_score(df, max_workers=8):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(compute_intent_score_for_row, [row for _, row in df.iterrows()]))
    return pd.Series(results, index=df.index)


def batch_score_intent(
    texts: List[str],
    labels: Optional[List[str]] = None,
    model_name: str = ZSC_MODEL_NAME,
    batch_size: int = 16
) -> List[Dict[str, float]]:
    """
    Batched intent scoring for throughput. Returns list of per-text dicts.
    """
    zsc = get_zero_shot_pipeline(model_name)
    use_labels = labels or INTENT_LABELS
    # outputs: List[Dict[str, float]] = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        res_list = zsc(sequences=chunk, candidate_labels=use_labels, multi_label=False)
        if isinstance(res_list, dict):
            res_list = [res_list]
        for res in res_list:
            scores = dict(zip(res["labels"], res["scores"]))
            for lbl in use_labels:
                scores.setdefault(lbl, 0.0)
            scores["S_intent"] = float(scores.get("genuine", 0.0))
            # outputs.append(scores)
    return scores["S_intent"]