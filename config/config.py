"""
Configuration file for the Job Recommendation System project.
Contains all constants, paths, and configurations needed across the project.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"

# Dataset file paths
JOBS_CSV_PATH = RAW_DATA_DIR / "preprocessed_jobs.csv"
RESUMES_CSV_PATH = RAW_DATA_DIR / "preprocessed_resumes.csv"

# Processed dataset file paths
PROCESSED_JOBS_PATH = PROCESSED_DATA_DIR / "jobs_cleaned.csv"
PROCESSED_RESUMES_PATH = PROCESSED_DATA_DIR / "resumes_cleaned.csv"
JOBS_EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "jobs_embeddings.npy"
RESUMES_EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "resumes_embeddings.npy"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_data.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_data.csv"

# Model paths
MODEL_PATH = MODELS_DIR / "recommendation_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index.bin"

# Configuration parameters
DATA_CLEANING_CONFIG = {
    # Jobs filtering
    "RELEVANT_CATEGORIES": ["INFORMATION-TECHNOLOGY", "HR", "FINANCE", "SALES", "BUSINESS-DEVELOPMENT"],
    "RELEVANT_JOB_KEYWORDS": ["job", "manager", "engineer", "scientist", "developer", "generalist", "analyst"],

    # Data validation
    "MIN_SALARY": 30000,
    "MAX_SALARY": 300000,
    "MAX_EXPERIENCE_YEARS": 30,
    "MIN_EXPERIENCE_YEARS": 0,

    # Missing data handling
    "MISSING_VALUE_THRESHOLD": 0.5,
}

# Feature engineering config
FEATURE_ENGINEERING_CONFIG = {
    "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
    "EMBEDDING_DIM": 384,
    "USE_TF_IDF": True,
    "MAX_FEATURES_TFIDF": 1000,
}

# Model training config
MODEL_CONFIG = {
    "RANDOM_SEED": 42,
    "TEST_SIZE": 0.2,
    "TRAIN_SIZE": 0.8,
    "CV_FOLDS": 5,

    # Random Forest parameters
    "RF_N_ESTIMATORS": 100,
    "RF_MAX_DEPTH": 20,
    "RF_MIN_SAMPLES_SPLIT": 5,
    "RF_MIN_SAMPLES_LEAF": 2,

    # XGBoost parameters
    "XGB_N_ESTIMATORS": 100,
    "XGB_MAX_DEPTH": 6,
    "XGB_LEARNING_RATE": 0.1,
    "XGB_SUBSAMPLE": 0.8,
}

# Evaluation config
EVALUATION_CONFIG = {
    "PRECISION_AT_K": [5, 10, 15],
    "R2_TARGET": 0.75,
    "PRECISION_AT_5_TARGET": 0.80,
}

# Streaming app config
STREAMLIT_CONFIG = {
    "MAX_RECOMMENDATIONS": 15,
    "DISPLAY_RECOMMENDATIONS": 10,
    "SIDEBAR_WIDTH": "small",
}

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
