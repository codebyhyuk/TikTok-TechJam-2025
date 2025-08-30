import pandas as pd
import numpy as np
import sys, os
import gzip, json
import re


def preprocess_review_metadata(df_review, df_meta):
    df_reviews_reduced = df_review[["user_id", "time", "rating", "text", "gmap_id"]].copy()
    df_metadata_reduced = df_meta[["gmap_id", "name", "category", "description"]].copy()

    df_merged = pd.merge(df_reviews_reduced, df_metadata_reduced, on="gmap_id", how="left")

    df_merged_processed = df_merged.rename(columns={"name":"business_name", "category":"business_category", "description":"business_description"})
    df_merged_processed["_id"] = df_merged_processed["user_id"].astype(str) + "_" + df_merged_processed["time"].astype(str)
    df_merged_processed = df_merged_processed.drop(columns=["user_id", "time", "gmap_id"])

    df_merged_drop_dup = df_merged_processed.drop_duplicates(subset=["_id"])
    df_merged_final = df_merged_drop_dup[~df_merged_drop_dup.isnull().any(axis=1)]
    df_merged_final["result"] = None
    df_result = df_merged_final.reset_index(drop=True).copy()
    return df_result


def preprocess_policy_C(df):
    condition = (df["text"].str.len()<20)

    df['result'] = np.where((condition & df["result"].isna()), 0, df["result"])
    return df


def preprocess_policy_D1(df):
    allowed_pattern = r'^[A-Za-z0-9\s.,!?;:\'\"()\[\]\{\}<>@#$%^&*_+=~`|\\/–—]*$'

    condition = ~(
        df['text'].str.match(allowed_pattern, na=False) &
        df['business_name'].str.match(allowed_pattern, na=False) &
        # df['business_category'].str.match(allowed_pattern, na=False)
        df['business_description'].str.match(allowed_pattern, na=False)
    )

    df['result'] = np.where(condition & df['result'].isna(), 0, df['result'])
    return df


def preprocess_policy_F(df):
    link_pattern = r"(https?://\S+|www\.\S+)"    
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"  
    phone_pattern = r"\b(?:\+?\d{1,3})?[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"  
    combined_pattern = f"({link_pattern}|{email_pattern}|{phone_pattern})"

    condition = (df['text'].str.contains(combined_pattern, regex=True))

    df['result'] = np.where((condition & df["result"].isna()), 0, df["result"])
    return df
