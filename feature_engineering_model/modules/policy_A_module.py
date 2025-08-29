
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel
import numpy as np
from scipy.spatial.distance import cosine
import os

def policy_A_similarity(file_path: str) -> list[float]:
    """
    This function takes a file path to a CSV file, processes the text data,
    trains an LDA model, and returns a list of cosine similarity scores
    between review and business documents for each row.
    The output is a list of floats between 0 and 1.
    """
    # Ensure NLTK data is downloaded
    for resource in ['punkt', 'stopwords', 'wordnet']:
        try:
            nltk.data.find(f'corpora/{resource}' if resource != 'punkt' else f'tokenizers/{resource}')
        except nltk.downloader.DownloadError:
            nltk.download(resource)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    # Data cleaning and feature engineering
    df['review_document'] = df['text'].astype(str)
    df['business_document'] = (df['business_name'].fillna('') + ' ' + 
                               df['business_category'].fillna('') + ' ' + 
                               df['business_description'].fillna(''))

    corpus = []
    for index, row in df.iterrows():
        corpus.append(row['review_document'])
        corpus.append(row['business_document'])

    # Text preprocessing
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        return [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]

    processed_corpus = [preprocess_text(doc) for doc in corpus]

    dictionary = corpora.Dictionary(processed_corpus)
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    # LDA model training
    lda_model = LdaModel(bow_corpus, num_topics=100, id2word=dictionary, passes=15)

    def get_lda_vector(text, lda_model, dictionary, preprocess_text_func):
        processed_text = preprocess_text_func(text)
        bow_vector = dictionary.doc2bow(processed_text)
        lda_vector = lda_model.get_document_topics(bow_vector, minimum_probability=0.0)
        dense_vector = np.zeros(lda_model.num_topics)
        for topic_num, prop_topic in lda_vector:
            dense_vector[topic_num] = prop_topic
        return dense_vector

    def calculate_cosine_similarity(vec1, vec2):
        # Handle zero vectors
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        return 1 - cosine(vec1, vec2)

    similarity_scores = []
    for index, row in df.iterrows():
        review_vector = get_lda_vector(row['review_document'], lda_model, dictionary, preprocess_text)
        business_vector = get_lda_vector(row['business_document'], lda_model, dictionary, preprocess_text)
        similarity_score = calculate_cosine_similarity(review_vector, business_vector)
        similarity_scores.append(similarity_score)

    return similarity_scores
