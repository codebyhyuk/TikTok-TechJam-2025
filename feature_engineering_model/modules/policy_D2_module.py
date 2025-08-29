
import pandas as pd
from textblob import TextBlob
from langdetect import detect, LangDetectException
from sentence_transformers import SentenceTransformer, util
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

# Load the model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_interpretability_score(text):
    """
    Calculates the interpretability score for a given text.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    # 1. Language Detection
    try:
        if detect(text) != 'en':
            return 0.0
    except LangDetectException:
        return 0.0

    # Using TextBlob for sentence tokenization and POS tagging
    try:
        blob = TextBlob(text)
    except Exception:
        return 0.0

    # 2. Coherence (Sentence Similarity)
    sentences = [str(s) for s in blob.sentences]
    if len(sentences) < 2:
        if len(blob.words) < 3:
            return 0.1
        else:
            return 0.5

    embeddings = model.encode(sentences, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)

    coherence_scores = []
    for i in range(len(sentences) - 1):
        coherence_scores.append(cosine_scores[i][i+1].item())

    if not coherence_scores:
        return 0.5

    avg_coherence = np.mean(coherence_scores)
    normalized_coherence = (avg_coherence + 1) / 2

    # 3. POS tagging for basic grammar check (as a penalty)
    pos_tags = [tag for word, tag in blob.tags]
    has_verb = any(tag.startswith('VB') for tag in pos_tags)
    has_noun = any(tag.startswith('NN') for tag in pos_tags)
    if not (has_verb and has_noun):
        normalized_coherence *= 0.5

    return normalized_coherence

def calculate_interpretability_scores_for_df(df: pd.DataFrame, n_workers: int) -> pd.Series:
    """
    Calculates the interpretability score for each review in the DataFrame.

    Args:
        df: The input DataFrame with a 'text' column.
        n_workers: The number of workers for parallel processing.

    Returns:
        A pandas Series with the interpretability scores.
    """
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(calculate_interpretability_score, df['text']))
        
    return pd.Series(results, index=df.index)

if __name__ == '__main__':
    # Example usage:
    file_path = '../data_gpt_labeler/final_data_labeled_1.csv'
    df = pd.read_csv(file_path)
    
    # For demonstration, using a smaller sample
    df_sample = df.head(100).copy()
    
    print("Running interpretability score calculation...")
    start_time = time.time()
    interpretability_scores = calculate_interpretability_scores_for_df(df_sample, n_workers=4)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    print(interpretability_scores)
