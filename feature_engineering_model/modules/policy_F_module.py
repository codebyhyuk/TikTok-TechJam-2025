# # policy_F_modular_processes.py
# from __future__ import annotations
# from typing import Dict, List, Optional, Tuple, Sequence
# import os
# import numpy as np
# import pandas as pd
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import multiprocessing as mp
# from transformers import pipeline

# # ---------------- Config ----------------
# INTENT_LABELS = ['genuine', 'spam', 'advertising', 'competitor attack', 'incentivize', 'mistaken identity']
# ZSC_MODEL_NAME = "facebook/bart-large-mnli"

# # ---------------- Per-process cache ----------------
# _ZSC_PIPELINE = None

# def _init_worker(model_name: str):
#     """
#     Runs ONCE per worker process; preloads the HF pipeline into a per-process global.
#     """
#     os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
#     global _ZSC_PIPELINE
#     if _ZSC_PIPELINE is None:
#         _ZSC_PIPELINE = pipeline(task="zero-shot-classification", model=model_name)

# # ---------------- Single-text scoring (original semantics) ----------------
# def get_zero_shot_pipeline(model_name: str = ZSC_MODEL_NAME):
#     global _ZSC_PIPELINE
#     if _ZSC_PIPELINE is None:
#         _ZSC_PIPELINE = pipeline(task="zero-shot-classification", model=model_name)
#     return _ZSC_PIPELINE

# def score_intent(
#     text: str,
#     labels: Optional[List[str]] = None,
#     model_name: str = ZSC_MODEL_NAME,
# ) -> Dict[str, float]:
#     """
#     Original semantics:
#       - returns per-label probs + S_intent = P('genuine')
#       - empty text -> all zeros + S_intent = 0.0
#     """
#     use_labels = labels or INTENT_LABELS

#     if not text or not str(text).strip():
#         base = {lbl: 0.0 for lbl in use_labels}
#         base["S_intent"] = 0.0
#         return base

#     zsc = get_zero_shot_pipeline(model_name)
#     res = zsc(sequences=str(text), candidate_labels=use_labels, multi_label=False)
#     scores = dict(zip(res["labels"], res["scores"]))
#     for lbl in use_labels:
#         scores.setdefault(lbl, 0.0)
#     scores["S_intent"] = float(scores.get("genuine", 0.0))
#     return scores

# # ---------------- Worker (returns dicts, aligned to idxs) ----------------
# def _score_chunk_in_proc(
#     idxs: Sequence[int],
#     texts: List[str],
#     labels: List[str],
#     micro_batch: int,
# ) -> Tuple[Sequence[int], List[Dict[str, float]]]:
#     """
#     Executes inside a worker process. Returns a list of dicts (one per idx),
#     preserving original semantics and order.
#     """
#     global _ZSC_PIPELINE
#     zsc = _ZSC_PIPELINE

#     use_labels = labels
#     L = len(idxs)
#     # Pre-fill with None; we'll write each position as we go.
#     local_results: List[Optional[Dict[str, float]]] = [None] * L

#     # Buffer for non-empty texts + their local positions in this chunk
#     buf_texts: List[str] = []
#     buf_pos: List[int] = []

#     def empty_result_dict() -> Dict[str, float]:
#         d = {lbl: 0.0 for lbl in use_labels}
#         d["S_intent"] = 0.0
#         return d

#     def flush():
#         if not buf_texts:
#             return
#         try:
#             res_list = zsc(sequences=buf_texts, candidate_labels=use_labels, multi_label=False)
#             if isinstance(res_list, dict):
#                 res_list = [res_list]
#             assert len(res_list) == len(buf_texts)
#             for r, pos in zip(res_list, buf_pos):
#                 scores = dict(zip(r["labels"], r["scores"]))
#                 for lbl in use_labels:
#                     scores.setdefault(lbl, 0.0)
#                 scores["S_intent"] = float(scores.get("genuine", 0.0))
#                 local_results[pos] = scores
#         except Exception:
#             # If batch fails, fill those positions with zeros
#             for pos in buf_pos:
#                 local_results[pos] = empty_result_dict()
#         finally:
#             buf_texts.clear()
#             buf_pos.clear()

#     # Build local results respecting empty-text semantics
#     for local_i, global_i in enumerate(idxs):
#         t = texts[global_i]
#         if (t is None) or (not str(t).strip()):
#             local_results[local_i] = empty_result_dict()
#         else:
#             buf_texts.append(str(t))
#             buf_pos.append(local_i)
#             if len(buf_texts) >= micro_batch:
#                 flush()
#     flush()

#     # Safety: fill any missing slots (shouldn't happen)
#     for i in range(L):
#         if local_results[i] is None:
#             local_results[i] = empty_result_dict()

#     return idxs, local_results  # list[Dict] aligned to `idxs`

# # ---------------- Public MP API (list-of-texts) ----------------
# def batch_score_intent_processes(
#     texts: List[str],
#     labels: Optional[List[str]] = None,
#     model_name: str = ZSC_MODEL_NAME,
#     max_workers: int = 2,
#     task_chunk_size: int = 256,
#     micro_batch: int = 16,
#     prefer_fork: bool = False,
# ) -> List[Dict[str, float]]:
#     """
#     Multiprocessing equivalent of your original batch_score_intent(...).
#     Returns: list of per-text dicts, same order as input `texts`.
#     """
#     n = len(texts)
#     use_labels = labels or INTENT_LABELS
#     results: List[Optional[Dict[str, float]]] = [None] * n

#     def chunk_indices(m: int, size: int):
#         return [list(range(s, min(s + size, m))) for s in range(0, m, size)]

#     index_chunks = chunk_indices(n, max(1, task_chunk_size))
#     ctx = mp.get_context("fork" if prefer_fork else "spawn")

#     with ProcessPoolExecutor(
#         max_workers=max_workers,
#         initializer=_init_worker,
#         initargs=(model_name,),
#         mp_context=ctx,
#     ) as ex:
#         futs = [
#             ex.submit(_score_chunk_in_proc, idxs, texts, use_labels, micro_batch)
#             for idxs in index_chunks
#         ]
#         for fut in as_completed(futs):
#             idxs, rows = fut.result()  # rows: list[dict] aligned to idxs
#             for local_i, global_i in enumerate(idxs):
#                 results[global_i] = rows[local_i]

#     # Final safety fill (shouldn’t be needed)
#     def empty_result_dict() -> Dict[str, float]:
#         d = {lbl: 0.0 for lbl in use_labels}
#         d["S_intent"] = 0.0
#         return d

#     return [r if r is not None else empty_result_dict() for r in results]

# # ---------------- Convenience wrapper returning Series(S_intent) ----------------
# def compute_policy_f_scores_processes(
#     df: pd.DataFrame,
#     text_col: str = "text",
#     labels: Optional[List[str]] = None,
#     model_name: str = ZSC_MODEL_NAME,
#     max_workers: int = 2,
#     task_chunk_size: int = 256,
#     micro_batch: int = 16,
#     prefer_fork: bool = False,
# ) -> pd.Series:
#     """
#     Returns a pandas.Series of S_intent (P('genuine')) in [0,1], aligned to df.index,
#     while preserving all original semantics under the hood.
#     """
#     if text_col not in df.columns:
#         raise KeyError(f"Column '{text_col}' not in DataFrame.")

#     use_labels = labels or INTENT_LABELS
#     texts = df[text_col].tolist()
#     dicts = batch_score_intent_processes(
#         texts=texts,
#         labels=use_labels,
#         model_name=model_name,
#         max_workers=max_workers,
#         task_chunk_size=task_chunk_size,
#         micro_batch=micro_batch,
#         prefer_fork=prefer_fork,
#     )
#     s = np.array([float(d.get("S_intent", 0.0)) for d in dicts], dtype=float)
#     return pd.Series(s, index=df.index, name="policy_F_S_intent")


# if __name__ == "__main__":
#     # Example usage guarded for multiprocessing
#     df = pd.read_csv('/Users/evan/Documents/Projects/TikTok-TechJam-2025/data_gpt_labeler/final_data_2.csv')
#     sample_df = df.sample(n=10, random_state=42).reset_index(drop=True).copy()

#     # print(sample_df["text"])
#     # print(sample_df["text"].iloc[9])

#     policy_g_score = compute_policy_f_scores_processes(sample_df)
#     print(policy_g_score)

# --------------------------------------------------------------------------------

# from __future__ import annotations
# from typing import Dict, List, Optional, Tuple, Sequence
# import os
# import platform
# import numpy as np
# import pandas as pd
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import multiprocessing as mp
# from transformers import pipeline

# # ---------------- Config ----------------
# INTENT_LABELS = ['genuine', 'spam', 'advertising', 'competitor attack', 'incentivize', 'mistaken identity']
# LABEL_DEFS = {
#     "genuine": "an authentic customer review relevant to the business",
#     "spam": "irrelevant or bot-like content not related to the business",
#     "advertising": "self-promotional content or marketing",
#     "competitor attack": "content intended to harm a competitor’s reputation",
#     "incentivize": "review influenced by rewards or discounts",
#     "mistaken identity": "review about a different business or product"
# }

# ZSC_MODEL_NAME = "facebook/bart-large-mnli"

# # ---------------- Per-process cache ----------------
# _ZSC_PIPELINE = None
# _ZSC_MODEL_NAME = None  # track which model the cache holds


# def _init_worker(model_name: str):
#     """
#     Runs ONCE per worker process; preloads the HF pipeline into a per-process global.
#     """
#     os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
#     global _ZSC_PIPELINE, _ZSC_MODEL_NAME
#     if (_ZSC_PIPELINE is None) or (_ZSC_MODEL_NAME != model_name):
#         _ZSC_PIPELINE = pipeline(task="zero-shot-classification", model=model_name)
#         _ZSC_MODEL_NAME = model_name


# # ---------------- Helpers ----------------
# def _ensure_genuine(labels: List[str]) -> List[str]:
#     """Ensure 'genuine' is present and first; keep original order for the rest."""
#     if "genuine" in labels:
#         return ["genuine"] + [l for l in labels if l != "genuine"]
#     else:
#         return ["genuine"] + list(labels)


# def _empty_result_dict(use_labels: List[str]) -> Dict[str, float]:
#     d = {lbl: 0.0 for lbl in use_labels}
#     d["S_intent"] = 0.0
#     return d


# def _compose_input(review_text: str, business_desc: Optional[str]) -> str:
#     """
#     Compose the classifier input. Preserve original semantics:
#     - If review_text is empty/blank -> caller will short-circuit to zeros
#       regardless of business_desc.
#     """
#     if business_desc and str(business_desc).strip():
#         return f"Business description: {business_desc}\nReview: {review_text}"
#     return str(review_text)


# # Single-text scoring (with optional business context) 
# def get_zero_shot_pipeline(model_name: str = ZSC_MODEL_NAME):
#     global _ZSC_PIPELINE, _ZSC_MODEL_NAME
#     if (_ZSC_PIPELINE is None) or (_ZSC_MODEL_NAME != model_name):
#         _ZSC_PIPELINE = pipeline(task="zero-shot-classification", model=model_name)
#         _ZSC_MODEL_NAME = model_name
#     return _ZSC_PIPELINE


# def score_intent(
#     text: str,
#     business_desc: Optional[str] = None,
#     labels: Optional[List[str]] = None,
#     model_name: str = ZSC_MODEL_NAME,
# ) -> Dict[str, float]:
#     """
#     Scores intent for a single review, optionally including a business description
#     for better context.

#     Semantics preserved:
#       - returns per-label probs + S_intent = P('genuine')
#       - empty TEXT -> all zeros + S_intent = 0.0 (even if business_desc exists)
#     """
#     use_labels = _ensure_genuine(labels or INTENT_LABELS)
#     label_texts = [f"{l} (meaning: {LABEL_DEFS.get(l, l)})" for l in labels]
    
#     if not text or not str(text).strip():
#         return _empty_result_dict(use_labels)

#     zsc = get_zero_shot_pipeline(model_name)
#     input_text = _compose_input(text, business_desc)
#     res = zsc(
#         sequences=input_text, 
#         candidate_labels=label_texts, 
#         multi_label=False,
#         hypothesis_template="Given the business context above, this customer review is {label}."
#     )

#     scores = dict(zip(res["labels"], res["scores"]))
#     for lbl in use_labels:
#         scores.setdefault(lbl, 0.0)
#     scores["S_intent"] = float(scores.get("genuine", 0.0))
#     return scores


# # ---------------- Worker (returns dicts, aligned to idxs) ----------------
# def _score_chunk_in_proc(
#     base_idx: int,
#     texts_chunk: List[str],
#     descs_chunk: Optional[List[Optional[str]]],
#     labels: List[str],
#     micro_batch: int,
# ) -> Tuple[Sequence[int], List[Dict[str, float]]]:
#     """
#     Executes inside a worker process. Receives only its slice of texts (and optional
#     business descriptions). Returns a list of dicts aligned to base_idx..base_idx+L-1.
#     """
#     global _ZSC_PIPELINE
#     zsc = _ZSC_PIPELINE
#     use_labels = labels
#     L = len(texts_chunk)

#     local_results: List[Optional[Dict[str, float]]] = [None] * L
#     buf_texts: List[str] = []
#     buf_pos: List[int] = []

#     def empty_result_dict() -> Dict[str, float]:
#         return _empty_result_dict(use_labels)

#     def flush():
#         if not buf_texts:
#             return
#         try:
#             res_list = zsc(sequences=buf_texts, candidate_labels=use_labels, multi_label=False)
#             if isinstance(res_list, dict):
#                 res_list = [res_list]
#             assert len(res_list) == len(buf_texts)
#             for r, pos in zip(res_list, buf_pos):
#                 scores = dict(zip(r["labels"], r["scores"]))
#                 for lbl in use_labels:
#                     scores.setdefault(lbl, 0.0)
#                 scores["S_intent"] = float(scores.get("genuine", 0.0))
#                 local_results[pos] = scores
#         except Exception:
#             # If batch fails, fill those positions with zeros
#             for pos in buf_pos:
#                 local_results[pos] = empty_result_dict()
#         finally:
#             buf_texts.clear()
#             buf_pos.clear()

#     # Build local results respecting empty-text semantics
#     for local_i, t in enumerate(texts_chunk):
#         if (t is None) or (not str(t).strip()):
#             local_results[local_i] = empty_result_dict()
#         else:
#             bd = descs_chunk[local_i] if descs_chunk is not None else None
#             composed = _compose_input(str(t), bd)
#             buf_texts.append(composed)
#             buf_pos.append(local_i)
#             if len(buf_texts) >= micro_batch:
#                 flush()
#     flush()

#     # Safety: fill any missing slots (shouldn't happen)
#     for i in range(L):
#         if local_results[i] is None:
#             local_results[i] = empty_result_dict()

#     idxs = list(range(base_idx, base_idx + L))
#     return idxs, local_results  # list[Dict] aligned to idxs


# # Public MP API (list-of-texts + optional list-of-descs) 
# def batch_score_intent_processes(
#     texts: List[str],
#     business_descs: Optional[List[Optional[str]]] = None,
#     labels: Optional[List[str]] = None,
#     model_name: str = ZSC_MODEL_NAME,
#     max_workers: int = 2,
#     task_chunk_size: int = 256,
#     micro_batch: int = 16,
#     prefer_fork: bool = False,
# ) -> List[Dict[str, float]]:
#     """
#     Multiprocessing batch scoring with optional per-row business descriptions.

#     Arguments:
#       texts: list[str] of reviews
#       business_descs: optional list[str|None], same length as texts, containing
#                       the business description for each row (or None)
#     Returns:
#       list[dict] aligned with `texts`, each containing label scores + S_intent.
#     """
#     n = len(texts)
#     use_labels = _ensure_genuine(labels or INTENT_LABELS)
#     results: List[Optional[Dict[str, float]]] = [None] * n

#     if business_descs is not None and len(business_descs) != n:
#         raise ValueError("business_descs must be None or the same length as texts")

#     def chunk_bounds(m: int, size: int):
#         return [(s, min(s + size, m)) for s in range(0, m, max(1, size))]

#     index_ranges = chunk_bounds(n, task_chunk_size)

#     # Guard: on macOS and many Linux distros with HF tokenizers, prefer "spawn"
#     start_method = "fork" if prefer_fork and platform.system() != "Darwin" else "spawn"
#     ctx = mp.get_context(start_method)

#     with ProcessPoolExecutor(
#         max_workers=max_workers,
#         initializer=_init_worker,
#         initargs=(model_name,),
#         mp_context=ctx,
#     ) as ex:
#         futs = []
#         for s, e in index_ranges:
#             futs.append(
#                 ex.submit(
#                     _score_chunk_in_proc,
#                     s,
#                     texts[s:e],                              # pass only the slice to the worker
#                     (business_descs[s:e] if business_descs is not None else None),
#                     use_labels,
#                     micro_batch,
#                 )
#             )

#         for fut in as_completed(futs):
#             try:
#                 idxs, rows = fut.result()  # rows: list[dict] aligned to idxs
#             except Exception:
#                 # Leave Nones for this chunk; we'll fill them with zeros below.
#                 continue

#             for local_i, global_i in enumerate(idxs):
#                 results[global_i] = rows[local_i]

#     # Final safety fill (in case a future failed entirely)
#     def empty_result_dict() -> Dict[str, float]:
#         return _empty_result_dict(use_labels)

#     return [r if r is not None else empty_result_dict() for r in results]


# # ---------------- Convenience wrapper returning Series(S_intent) ----------------
# def compute_policy_f_scores_processes(
#     df: pd.DataFrame,
#     text_col: str = "text",
#     business_desc_col: Optional[str] = "business_description",
#     labels: Optional[List[str]] = None,
#     model_name: str = ZSC_MODEL_NAME,
#     max_workers: int = 2,
#     task_chunk_size: int = 256,
#     micro_batch: int = 16,
#     prefer_fork: bool = False,
# ) -> pd.Series:
#     """
#     Returns a pandas.Series of S_intent (P('genuine')) in [0,1], aligned to df.index.

#     If `business_desc_col` is provided and exists in df, each row's business
#     description will be included as context for the model. If the review text is
#     empty/blank, scores are zeros regardless of business description (original semantics).
#     """
#     if text_col not in df.columns:
#         raise KeyError(f"Column '{text_col}' not in DataFrame.")

#     use_labels = _ensure_genuine(labels or INTENT_LABELS)
#     texts = df[text_col].tolist()

#     descs = None
#     if business_desc_col is not None and (business_desc_col in df.columns):
#         # Accept strings or None/NaN per row
#         descs = df[business_desc_col].where(pd.notnull(df[business_desc_col]), None).tolist()

#     dicts = batch_score_intent_processes(
#         texts=texts,
#         business_descs=descs,
#         labels=use_labels,
#         model_name=model_name,
#         max_workers=max_workers,
#         task_chunk_size=task_chunk_size,
#         micro_batch=micro_batch,
#         prefer_fork=prefer_fork,
#     )
#     s = np.array([float(d.get("S_intent", 0.0)) for d in dicts], dtype=float)
#     return pd.Series(s, index=df.index, name="policy_F_S_intent")


# if __name__ == "__main__":
#     # Example usage (guarded for multiprocessing)
#     df = pd.read_csv('/Users/evan/Documents/Projects/TikTok-TechJam-2025/data_gpt_labeler/final_data_2.csv')
#     sample_df = df.sample(n=10, random_state=42).reset_index(drop=True).copy()
#     # Ensure you have a 'business_desc' column if you want to include context.
#     print(sample_df["text"].iloc[2])
#     print(sample_df["business_description"].iloc[2])
#     policy_f_score = compute_policy_f_scores_processes(
#         sample_df,
#         text_col="text",
#         business_desc_col="business_description",
#         max_workers=2,
#         task_chunk_size=128,
#         micro_batch=16,
#         prefer_fork=False,  # keep False on macOS to avoid fork-related issues
#     )
#     print(policy_f_score)
    
# --------------------------------------------------------------------------------

# policy_F_crossencoder_processes.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Sequence
import os
import platform
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# sentence-transformers CrossEncoder
from sentence_transformers import CrossEncoder

# ---------------- Config ----------------
INTENT_LABELS = [
    "genuine",
    "spam",
    "advertising",
    "competitor attack"
]
ZSC_MODEL_NAME = "cross-encoder/nli-deberta-v3-large"

# You can adjust this template; keep {label} in it.
HYPOTHESIS_TEMPLATE = (
    "Given the business context above, this customer review is {label}."
)

# ---------------- Per-process cache ----------------
_XENCODER = None
_XENCODER_NAME = None
_ENT_IDX = None  # index of 'entailment' in the model's output order


def _find_entailment_index(model: CrossEncoder) -> int:
    """
    Determine which output index corresponds to 'entailment'.
    This inspects the underlying HF model's id2label/label2id mapping.
    Fallback to 1 (common for many NLI heads) if not available.
    """
    try:
        id2label = getattr(model.model.config, "id2label", None)
        if isinstance(id2label, dict) and len(id2label) >= 3:
            # normalize labels, e.g., 'ENTAILMENT' -> 'entailment'
            norm = {int(i): str(lbl).lower() for i, lbl in id2label.items()}
            for i, name in norm.items():
                if "entail" in name:
                    return int(i)
        # Some configs expose label2id
        label2id = getattr(model.model.config, "label2id", None)
        if isinstance(label2id, dict) and len(label2id) >= 3:
            for name, i in label2id.items():
                if "entail" in str(name).lower():
                    return int(i)
    except Exception:
        pass
    # Sensible default in sentence-transformers examples is often index 1
    return 1


def _init_worker(model_name: str, device: str = "cpu"):
    """
    Runs ONCE per worker process; loads the CrossEncoder into a per-process global.
    device: 'cpu' or 'cuda' (note: if using 'cuda', prefer max_workers=1)
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    global _XENCODER, _XENCODER_NAME, _ENT_IDX
    if (_XENCODER is None) or (_XENCODER_NAME != model_name):
        _XENCODER = CrossEncoder(model_name, device=device)
        _XENCODER_NAME = model_name
        _ENT_IDX = _find_entailment_index(_XENCODER)


# ---------------- Helpers ----------------
def _ensure_genuine(labels: List[str]) -> List[str]:
    """Ensure 'genuine' is present and first; keep original order for the rest."""
    if "genuine" in labels:
        return ["genuine"] + [l for l in labels if l != "genuine"]
    else:
        return ["genuine"] + list(labels)


def _empty_result_dict(use_labels: List[str]) -> Dict[str, float]:
    d = {lbl: 0.0 for lbl in use_labels}
    d["S_intent"] = 0.0
    return d


def _compose_input(review_text: str, business_desc: Optional[str]) -> str:
    """
    Compose the NLI premise. If review_text is empty, caller short-circuits to zeros.
    """
    if business_desc and str(business_desc).strip():
        return f"Business description: {business_desc}\nReview: {review_text}"
    return str(review_text)


def _softmax_over_labels(entail_scores: np.ndarray) -> np.ndarray:
    """
    Apply softmax across labels (last axis).
    Input: [K] entailment scores (logits or probs monotonic in entailment)
    Output: [K] probabilities summing to 1.
    """
    x = entail_scores - np.max(entail_scores)
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)


# ---------------- Single-text scoring ----------------
def score_intent(
    text: str,
    business_desc: Optional[str] = None,
    labels: Optional[List[str]] = None,
    model_name: str = ZSC_MODEL_NAME,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Cross-encoder NLI scoring for a single review.
      - premise = business_desc + review
      - hypotheses = HYPOTHESIS_TEMPLATE.format(label=...)
      - probability per label computed by softmax over entailment scores
      - empty text => all zeros + S_intent=0.0
    """
    use_labels = _ensure_genuine(labels or INTENT_LABELS)

    if not text or not str(text).strip():
        return _empty_result_dict(use_labels)

    global _XENCODER, _XENCODER_NAME, _ENT_IDX
    if (_XENCODER is None) or (_XENCODER_NAME != model_name):
        _init_worker(model_name, device=device)

    premise = _compose_input(text, business_desc)
    hyps = [HYPOTHESIS_TEMPLATE.format(label=lbl) for lbl in use_labels]

    pairs = [(premise, h) for h in hyps]  # K pairs
    # # Predict class probabilities; we want probability for 'entailment' index
    # probs = _XENCODER.predict(pairs, apply_softmax=True)  # shape: [K, C]
    # ent = np.array([p[_ENT_IDX] for p in probs], dtype=float)  # [K]
    # label_probs = _softmax_over_labels(ent)  # normalize across labels
    # single-text path (and similarly in the worker):
    probs = _XENCODER.predict(pairs, apply_softmax=False)  # logits, shape [K, C]
    ent_logits = np.array([logits[_ENT_IDX] for logits in probs], dtype=float)
    label_probs = _softmax_over_labels(ent_logits)         # softmax across labels


    out = {lbl: float(p) for lbl, p in zip(use_labels, label_probs)}
    out["S_intent"] = out.get("genuine", 0.0)
    return out


# ---------------- Worker (returns dicts, aligned to idxs) ----------------
def _score_chunk_in_proc(
    base_idx: int,
    texts_chunk: List[str],
    descs_chunk: Optional[List[Optional[str]]],
    labels: List[str],
    micro_batch: int,
) -> Tuple[Sequence[int], List[Dict[str, float]]]:
    """
    Executes inside a worker process with the CrossEncoder cached in this process.
    For efficiency, we process 'micro_batch' texts at a time; each text expands
    to K hypotheses, producing M*K pairs per forward pass.
    """
    global _XENCODER, _ENT_IDX
    xenc = _XENCODER
    ent_idx = _ENT_IDX
    use_labels = labels
    L = len(texts_chunk)
    K = len(use_labels)

    local_results: List[Optional[Dict[str, float]]] = [None] * L

    def empty_result_dict() -> Dict[str, float]:
        return _empty_result_dict(use_labels)

    # Process in micro-batches of texts
    for start in range(0, L, max(1, micro_batch)):
        end = min(start + micro_batch, L)
        sub = texts_chunk[start:end]
        sub_desc = descs_chunk[start:end] if descs_chunk is not None else [None] * (end - start)

        # Build premise-hypothesis pairs for this micro-batch
        pairs: List[Tuple[str, str]] = []
        grid_positions: List[Tuple[int, int]] = []  # (local_i, label_i)

        # First: create placeholders and handle empty-text rows
        need_predict = [False] * (end - start)
        for i, t in enumerate(sub):
            if (t is None) or (not str(t).strip()):
                local_results[start + i] = empty_result_dict()
            else:
                need_predict[i] = True

        # Build pairs for rows that need prediction
        for i, need in enumerate(need_predict):
            if not need:
                continue
            premise = _compose_input(str(sub[i]), sub_desc[i])
            for j, lbl in enumerate(use_labels):
                pairs.append((premise, HYPOTHESIS_TEMPLATE.format(label=lbl)))
                grid_positions.append((i, j))

        if not pairs:
            continue

        try:
            # Predict per pair → get per-class probabilities
            probs = xenc.predict(pairs, apply_softmax=True)  # [Npairs, C]
            probs = np.asarray(probs, dtype=float)
            entail = probs[:, ent_idx]  # [Npairs]

            # Aggregate back to per-text [K] arrays
            # Initialize with very negative for stability; we'll fill real ones.
            stitched = {i: np.full(K, -1e9, dtype=float) for i, need in enumerate(need_predict) if need}
            for (i, j), v in zip(grid_positions, entail):
                stitched[i][j] = v

            # Now softmax across labels per text and write results
            for i in range(end - start):
                if not need_predict[i]:
                    continue
                ent_vec = stitched[i]  # [K] entailment proxies
                label_probs = _softmax_over_labels(ent_vec)
                d = {lbl: float(p) for lbl, p in zip(use_labels, label_probs)}
                d["S_intent"] = d.get("genuine", 0.0)
                local_results[start + i] = d

        except Exception:
            # If anything fails, fill these rows with zeros (graceful degradation)
            for i in range(start, end):
                if local_results[i] is None:
                    local_results[i] = empty_result_dict()

    # Safety: fill any missing slots (shouldn't happen)
    for i in range(L):
        if local_results[i] is None:
            local_results[i] = empty_result_dict()

    idxs = list(range(base_idx, base_idx + L))
    return idxs, local_results  # list[Dict] aligned to idxs


# ---------------- Public MP API (list-of-texts + optional list-of-descs) ----------------
def batch_score_intent_processes(
    texts: List[str],
    business_descs: Optional[List[Optional[str]]] = None,
    labels: Optional[List[str]] = None,
    model_name: str = ZSC_MODEL_NAME,
    device: str = "cpu",
    max_workers: int = 2,
    task_chunk_size: int = 256,
    micro_batch: int = 16,
    prefer_fork: bool = False,
) -> List[Dict[str, float]]:
    """
    Multiprocessing batch scoring with CrossEncoder and optional per-row business descriptions.
    Returns list[dict] aligned to `texts`, each containing label probs + S_intent.
    """
    n = len(texts)
    use_labels = _ensure_genuine(labels or INTENT_LABELS)
    results: List[Optional[Dict[str, float]]] = [None] * n

    if business_descs is not None and len(business_descs) != n:
        raise ValueError("business_descs must be None or the same length as texts")

    def chunk_bounds(m: int, size: int):
        return [(s, min(s + size, m)) for s in range(0, m, max(1, size))]

    index_ranges = chunk_bounds(n, task_chunk_size)

    # Guard: CUDA across multiple processes is tricky. If device == 'cuda', prefer single worker.
    if device == "cuda" and max_workers != 1:
        raise ValueError("When using device='cuda', set max_workers=1 to avoid GPU contention.")

    start_method = "fork" if (prefer_fork and platform.system() != "Darwin") else "spawn"
    ctx = mp.get_context(start_method)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(model_name, device),
        mp_context=ctx,
    ) as ex:
        futs = []
        for s, e in index_ranges:
            futs.append(
                ex.submit(
                    _score_chunk_in_proc,
                    s,
                    texts[s:e],
                    (business_descs[s:e] if business_descs is not None else None),
                    use_labels,
                    micro_batch,
                )
            )
        for fut in as_completed(futs):
            try:
                idxs, rows = fut.result()
            except Exception:
                # Leave Nones; we'll fill with zeros below.
                continue
            for local_i, global_i in enumerate(idxs):
                results[global_i] = rows[local_i]

    def empty_result_dict() -> Dict[str, float]:
        return _empty_result_dict(use_labels)

    return [r if r is not None else empty_result_dict() for r in results]


# ---------------- Convenience wrapper returning Series(S_intent) ----------------
def compute_policy_f_scores_processes(
    df: pd.DataFrame,
    text_col: str = "text",
    business_desc_col: Optional[str] = "business_description",
    labels: Optional[List[str]] = None,
    model_name: str = ZSC_MODEL_NAME,
    device: str = "cpu",
    max_workers: int = 2,
    task_chunk_size: int = 256,
    micro_batch: int = 16,
    prefer_fork: bool = False,
) -> pd.Series:
    """
    Returns a pandas.Series of S_intent (P('genuine')) in [0,1], aligned to df.index.
    """
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not in DataFrame.")

    use_labels = _ensure_genuine(labels or INTENT_LABELS)
    texts = df[text_col].tolist()

    descs = None
    if business_desc_col is not None and (business_desc_col in df.columns):
        descs = df[business_desc_col].where(pd.notnull(df[business_desc_col]), None).tolist()

    dicts = batch_score_intent_processes(
        texts=texts,
        business_descs=descs,
        labels=use_labels,
        model_name=model_name,
        device=device,
        max_workers=max_workers,
        task_chunk_size=task_chunk_size,
        micro_batch=micro_batch,
        prefer_fork=prefer_fork,
    )
    s = np.array([float(d.get("S_intent", 0.0)) for d in dicts], dtype=float)
    return pd.Series(s, index=df.index, name="policy_F_S_intent")


if __name__ == "__main__":
    # Example (CPU; if using GPU set device='cuda' and max_workers=1)
    # df = pd.read_csv("/Users/evan/Documents/Projects/TikTok-TechJam-2025/data_gpt_labeler/final_data_4.csv")
    # sample_df = df.sample(n=10, random_state=42).reset_index(drop=True).copy()
    sample_df = pd.DataFrame({
    "text": [
        "Latte art was gorgeous and the muffin was still warm.",
        "Use code COFFEE10 for discounts at my shop!",
        "Don't waste your time here—RivalBrew is way better.",
        ""
    ],
    "business_description": [
        "Artisanal café offering latte art and fresh daily bakes.",
        "Community coffee shop serving espresso, tea, and pastries.",
        "BrewBox coffeehouse known for seasonal drinks and light bites.",
        "Independent roastery with a small espresso bar."
    ]
})

    
    print(sample_df["text"])
    print(sample_df["business_description"])
    scores = compute_policy_f_scores_processes(
        sample_df,
        text_col="text",
        business_desc_col="business_description",
        model_name=ZSC_MODEL_NAME,
        device="cpu",              # or "cuda" with max_workers=1
        max_workers=2,             # keep >1 only for CPU
        task_chunk_size=128,
        micro_batch=16,
        prefer_fork=False,
    )
    print(scores)
    
