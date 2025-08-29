import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
from scipy.spatial.distance import cosine
import os

def download_nltk_data():
    """Downloads necessary NLTK data if not already present."""
    packages = ['punkt', 'stopwords', 'wordnet']
    for package in packages:
        try:
            if package == 'punkt':
                nltk.data.find(f'tokenizers/{package}')
            else:
                nltk.data.find(f'corpora/{package}')
        except LookupError:
            nltk.download(package)

def policy_A_feature_generation(df: pd.DataFrame) -> pd.Series:
    """
    Modularized process from policy_A.ipynb to generate a similarity feature.

    This function takes a DataFrame, processes the text data to create
    review and business documents, trains an LDA model on the corpus,
    and then calculates the cosine similarity between the LDA vectors
    of the review and business documents for each row.

    Args:
        df (pd.DataFrame): Input DataFrame with 'text', 'business_name',
                           'business_category', and 'business_description' columns.

    Returns:
        pd.Series: A 1-by-n pandas Series where n is the number of rows
                   in the input DataFrame. Each element is a 0-1 scale
                   feature representing the similarity score.
    """
    download_nltk_data()

    df_processed = df.copy()
    df_processed['review_document'] = df_processed['text'].astype(str)
    df_processed['business_document'] = (df_processed['business_name'].fillna('') + ' ' +
                                       df_processed['business_category'].fillna('') + ' ' +
                                       df_processed['business_description'].fillna(''))

    corpus = []
    for index, row in df_processed.iterrows():
        corpus.append(row['review_document'])
        corpus.append(row['business_document'])

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        return [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]

    processed_corpus = [preprocess_text(doc) for doc in corpus]

    dictionary = corpora.Dictionary(processed_corpus)
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    lda_model = LdaModel(bow_corpus, num_topics=100, id2word=dictionary, passes=15)

    def get_lda_vector(text):
        processed_text = preprocess_text(text)
        bow_vector = dictionary.doc2bow(processed_text)
        lda_vector = lda_model.get_document_topics(bow_vector, minimum_probability=0.0)
        dense_vector = np.zeros(lda_model.num_topics)
        for topic_num, prop_topic in lda_vector:
            dense_vector[topic_num] = prop_topic
        return dense_vector

    def calculate_cosine_similarity(vec1, vec2):
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        similarity = 1 - cosine(vec1, vec2)
        return similarity if not np.isnan(similarity) else 0.0

    similarity_scores = []
    for index, row in df_processed.iterrows():
        review_vector = get_lda_vector(row['review_document'])
        business_vector = get_lda_vector(row['business_document'])
        score = calculate_cosine_similarity(review_vector, business_vector)
        similarity_scores.append(score)

    return pd.Series(similarity_scores)

def policy_A_feature_generation_v2(df: pd.DataFrame, review: pd.Series) -> float:
    """
    Trains an LDA model on a DataFrame and uses it to calculate a similarity score
    for a single review row.

    The function processes the text data from the training DataFrame to create a
    corpus, trains an LDA model, and then calculates the cosine similarity
    between the LDA vectors of the review and business documents for a single
    provided review row.

    Args:
        df (pd.DataFrame): The training DataFrame with 'text', 'business_name',
                           'business_category', and 'business_description' columns.
        review (pd.Series): A single row (as a pandas Series) representing the
                            review for which to calculate the similarity score.

    Returns:
        float: A 0-1 scale feature representing the similarity score for the
               provided review. Returns 0.0 if any vector is empty.
    """
    # Download necessary NLTK data
    download_nltk_data()
    
    # --- Step 1: Preprocessing and Training the LDA Model on the entire DataFrame ---
    print("Training LDA model on the provided DataFrame...")
    
    df_processed = df.copy()
    df_processed['review_document'] = df_processed['text'].astype(str)
    df_processed['business_document'] = (
        df_processed['business_name'].fillna('') + ' ' +
        df_processed['business_category'].fillna('') + ' ' +
        df_processed['business_description'].fillna('')
    )

    corpus = []
    for _, row in df_processed.iterrows():
        corpus.append(row['review_document'])
        corpus.append(row['business_document'])

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        """Helper function to preprocess text for LDA."""
        tokens = word_tokenize(text.lower())
        return [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]

    processed_corpus = [preprocess_text(doc) for doc in corpus]
    
    if not processed_corpus:
        print("Warning: Training corpus is empty. Cannot train LDA model.")
        return 0.0

    dictionary = corpora.Dictionary(processed_corpus)
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    # Train the LDA model
    lda_model = LdaModel(bow_corpus, num_topics=100, id2word=dictionary, passes=15)
    print("LDA model training complete.")

    # --- Step 2: Generating LDA vectors and calculating similarity for the single review ---
    
    def get_lda_vector(text):
        """Helper function to get LDA vector for a given text."""
        processed_text = preprocess_text(text)
        bow_vector = dictionary.doc2bow(processed_text)
        lda_vector = lda_model.get_document_topics(bow_vector, minimum_probability=0.0)
        dense_vector = np.zeros(lda_model.num_topics)
        for topic_num, prop_topic in lda_vector:
            dense_vector[topic_num] = prop_topic
        return dense_vector

    def calculate_cosine_similarity(vec1, vec2):
        """Helper function to calculate cosine similarity between two vectors."""
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        similarity = 1 - cosine(vec1, vec2)
        return similarity if not np.isnan(similarity) else 0.0

    print("Calculating similarity score for the new review...")

    # Process the single review row
    review_document = review['text']
    business_document = (
        review['business_name'].fillna('') + ' ' +
        review['business_category'].fillna('') + ' ' +
        review['business_description'].fillna('')
    )

    # Get LDA vectors for the review and business documents
    review_vector = get_lda_vector(str(review_document))
    business_vector = get_lda_vector(str(business_document))

    # Calculate and return the similarity score
    score = calculate_cosine_similarity(review_vector, business_vector)
    
    print(f"Similarity score: {score:.4f}")
    return score

if __name__ == '__main__':
    # Example usage:
    # Assumes the script is run from the project root directory
    file_path = 'data_gpt_labeler/final_data_labeled_1.csv'
    
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}. Make sure you are running this script from the project root.")
    else:
        df = pd.read_csv(file_path)
        # Using a smaller sample for a quick test to avoid long processing time
        df_sample = df.head(10).copy() 
        
        print(f"Loading data from: {file_path}")
        print(f"Processing a sample of {len(df_sample)} rows...")
        
        similarity_series = policy_A_feature_generation_v2(df_sample, df.sample(1, random_state=42))
        
        print("\nGenerated similarity scores:")
        print(similarity_series)