from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def roberta_sentiment_score(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).numpy()[0]
    p_neg, p_neu, p_pos = probs
    s_text = p_pos - p_neg
    if p_neu > 0.75:
        s_prime = 0.5
    else:
        s_prime = (s_text + 1) / 2
    return s_prime


def normalize_rating(rating: int):
    return (rating - 1) / 4


def consistency_score(text_score: float, rating_score: float):
    return 1 - abs(text_score - rating_score)


def compute_consistency_for_row(row):
    text_score = roberta_sentiment_score(row['text'])
    rating_score = normalize_rating(row['rating'])
    return consistency_score(text_score, rating_score)


def compute_consistency_scores(df, max_workers=8):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(compute_consistency_for_row, [row for _, row in df.iterrows()]))
    return pd.Series(results, index=df.index)