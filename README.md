# TikTok TechJam 2025

A project for the TikTok TechJam 2025 Hackathon. This project aims to classify users based on their data.

## Project Overview

This project is divided into three main parts:
1.  **Data Labeling:** Using GPT to label the raw data based on a defined policy.
2.  **Feature Engineering:** Extracting meaningful features from the labeled data using various policies.
3.  **Classification:** Training and evaluating different classification models on the engineered features.

## Repository Structure

```
.
├── classification_model/       # Jupyter notebooks for classification models (MLP, RF, SVM, XGB)
├── data_gpt_labeler/           # Scripts and notebooks for data labeling with GPT
│   ├── filtered_datasets/      # Filtered datasets
│   └── labeled_datasets/       # GPT-labeled datasets
├── feature_engineering_model/  # Jupyter notebooks for feature engineering
│   ├── featured_datasets/      # Datasets with engineered features
│   └── modules/                # Python modules for feature engineering policies
├── images/                     # Images for the repository
│   └── Model Architecture.png  # Model architecture diagram
├── .gitignore                  # Git ignore file
├── final_data_featured_filtered.csv # Final dataset for classification
├── final_lda_train_data.csv    # Train data for LDA model
├── final_ml_pipeline.ipynb     # Jupyter notebook for the final ML pipeline
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd TikTok-TechJam-2025
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up OpenAI API Key:**
    The data labeling script for the generation of training dataset uses the OpenAI API. You need to set up your API key as an environment variable:
    ```bash
    export OPENAI_API_KEY='your-api-key'
    ```

## How to Reproduce Results

The project is structured as a pipeline. You can reproduce the results by running the steps in the following order:

1.  **Data Labeling:**
    - Navigate to the `data_gpt_labeler` directory.
    - Run the `data_preprocessing.ipynb` notebook, based on user specified dataset from [link](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/) to preprocess the data.
    - Run the `data_gpt_labeler_v2.py` script to label the data using GPT. This will generate the labeled datasets in the `labeled_datasets` directory.
    - *may require some absolute paths be updated with relative paths*

2.  **Feature Engineering:**
    - Navigate to the `feature_engineering_model` directory.
    - Run the `data_feature_extraction.ipynb` notebook. This notebook will use the modules in the `modules` directory to generate features from the labeled data. The resulting datasets with engineered features will be saved in the `featured_datasets` directory.

3.  **Classification:**
    - Navigate to the `classification_model` directory.
    - The `final_data_featured_filtered.csv` file is the final dataset used for classification.
    - You can run any of the notebooks (`MLP.ipynb`, `RF.ipynb`, `SVM.ipynb`, `XGB.ipynb`) to train and evaluate the corresponding classification model.

## Team Member Contribution
Lee Hyunseung

Park Yumin

Yoon Hyukjin