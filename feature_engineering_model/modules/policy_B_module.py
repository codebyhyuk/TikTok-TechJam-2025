
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
import math

def policy_B_specificity_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the specificity score for each review in the dataframe.

    Args:
        df: A pandas DataFrame containing a 'text' column with review texts.

    Returns:
        A pandas Series containing the specificity score for each review,
        on a scale of 0-1.
    """
    if df.empty:
        return pd.Series(dtype=float)

    # 1. Download necessary NLTK packages
    nltk_resources = {
        "punkt": "tokenizers/punkt",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "maxent_ne_chunker": "chunkers/maxent_ne_chunker",
        "words": "corpora/words"
    }
    for resource, path in nltk_resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource)

    # 2. Fit TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_vectorizer.fit(df['text'].fillna(''))

    # 3. Define feature functions
    def lexical_richness(text):
        if not isinstance(text, str) or not text:
            return 0
        tokens = word_tokenize(text)
        if not tokens:
            return 0
        return len(set(tokens)) / len(tokens)

    def named_entity_density(text):
        if not isinstance(text, str) or not text:
            return 0
        tokens = word_tokenize(text)
        if not tokens:
            return 0
        tagged_tokens = pos_tag(tokens)
        chunks = ne_chunk(tagged_tokens)
        named_entities = sum(1 for chunk in chunks if hasattr(chunk, 'label'))
        return named_entities / len(tokens)

    def get_average_tfidf_single(text, vectorizer):
        if not isinstance(text, str) or not text:
            return 0
        tfidf_matrix = vectorizer.transform([text])
        return tfidf_matrix.mean()

    def review_length(text):
        return len(text) if isinstance(text, str) else 0

    # 4. Apply feature functions to a copy to avoid SettingWithCopyWarning
    df_features = pd.DataFrame(index=df.index)
    df_features['lexical_richness'] = df['text'].apply(lexical_richness)
    df_features['named_entity_density'] = df['text'].apply(named_entity_density)
    df_features['tfidf_score'] = df['text'].apply(lambda x: get_average_tfidf_single(x, tfidf_vectorizer))
    df_features['review_length'] = df['text'].apply(review_length)

    # 5. Normalize features
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    max_len = df_features['review_length'].max()
    log_denominator = math.log(1 + max_len) if max_len > 0 else 1

    def log_scale(x):
        if log_denominator == 0: return 0
        return math.log(1 + x) / log_denominator

    df_features['lexical_richness_norm'] = df_features['lexical_richness'].apply(sigmoid)
    df_features['named_entity_density_norm'] = df_features['named_entity_density'].apply(sigmoid)
    df_features['tfidf_score_norm'] = df_features['tfidf_score'].apply(sigmoid)
    df_features['review_length_norm'] = df_features['review_length'].apply(log_scale)

    # 6. Calculate specificity score
    weights = {
        'lexical_richness_norm': 0.2,
        'named_entity_density_norm': 0.4,
        'tfidf_score_norm': 0.2,
        'review_length_norm': 0.2
    }

    specificity_score = (
        df_features['lexical_richness_norm'] * weights['lexical_richness_norm'] +
        df_features['named_entity_density_norm'] * weights['named_entity_density_norm'] +
        df_features['tfidf_score_norm'] * weights['tfidf_score_norm'] +
        df_features['review_length_norm'] * weights['review_length_norm']
    )

    return specificity_score
