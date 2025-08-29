
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor

def download_nltk_resources():
    """Download necessary NLTK resources."""
    resources = ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)

def lexical_richness(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        if len(tokens) == 0:
            return 0
        return len(set(tokens)) / len(tokens)
    return 0

def named_entity_density(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        if len(tokens) == 0:
            return 0
        tagged_tokens = pos_tag(tokens)
        chunks = ne_chunk(tagged_tokens)
        named_entities = 0
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                named_entities += 1
        return named_entities / len(tokens)
    return 0

def get_average_tfidf_single(text, vectorizer):
    if isinstance(text, str):
        tfidf_matrix = vectorizer.transform([text])
        return tfidf_matrix.mean()
    return 0

def review_length(text):
    if isinstance(text, str):
        return len(text)
    return 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log_scale(x):
    return math.log(1 + x) / math.log(1 + 2000)

def process_row(row, vectorizer, weights):
    """Process a single row of the DataFrame to calculate the specificity score."""
    text = row['text']
    
    # Calculate features
    lr = lexical_richness(text)
    ned = named_entity_density(text)
    tfidf = get_average_tfidf_single(text, vectorizer)
    rl = review_length(text)
    
    # Normalize features
    lr_norm = sigmoid(lr)
    ned_norm = sigmoid(ned)
    tfidf_norm = sigmoid(tfidf)
    rl_norm = log_scale(rl)
    
    # Calculate specificity score
    specificity_score = (
        lr_norm * weights['lexical_richness_norm'] +
        ned_norm * weights['named_entity_density_norm'] +
        tfidf_norm * weights['tfidf_score_norm'] +
        rl_norm * weights['review_length_norm']
    )
    
    return specificity_score

def calculate_specificity_score(df: pd.DataFrame, n_workers: int) -> pd.Series:
    """
    Calculates the specificity score for each review in the DataFrame.

    Args:
        df: The input DataFrame with a 'text' column.
        n_workers: The number of workers for parallel processing.

    Returns:
        A pandas Series with the specificity scores.
    """
    download_nltk_resources()
    
    # Fit the TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_vectorizer.fit(df['text'].fillna(''))
    
    weights = {
        'lexical_richness_norm': 0.2,
        'named_entity_density_norm': 0.4,
        'tfidf_score_norm': 0.2,
        'review_length_norm': 0.2
    }
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(lambda row: process_row(row, tfidf_vectorizer, weights), [row for _, row in df.iterrows()]))
        
    return pd.Series(results, index=df.index)

if __name__ == '__main__':
    # Example usage:
    file_path = '../data_gpt_labeler/final_data_labeled_1.csv'
    df = pd.read_csv(file_path)
    
    # For demonstration, using a smaller sample
    df_sample = df.sample(n=100, random_state=42).copy()
    
    specificity_scores = calculate_specificity_score(df_sample, n_workers=4)
    print(specificity_scores)
