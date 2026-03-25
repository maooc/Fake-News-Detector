"""
Pytest configuration and shared fixtures.
"""

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Add src to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_real_texts():
    """Sample real news texts for testing."""
    return [
        "The government announced new economic policies today.",
        "Scientists discovered a new species in the rainforest.",
        "The local team won the championship game yesterday.",
    ]


@pytest.fixture
def sample_fake_texts():
    """Sample fake news texts for testing."""
    return [
        "SHOCKING: Aliens control the government secretly!",
        "Doctors don't want you to know this miracle cure!",
        "Celebrity exposed as a lizard person from space!",
    ]


@pytest.fixture
def mini_dataset(tmp_path):
    """Create a minimal dataset for training tests."""
    real_texts = [
        "The government announced new policies today in Washington.",
        "Scientists discovered species in the Amazon rainforest region.",
        "Local team won championship after hard fought match.",
        "New hospital opened downtown to serve community needs.",
    ]
    fake_texts = [
        "SHOCKING secret documents reveal aliens control government!",
        "Miracle cure doctors don't want you to know about!!!",
        "Celebrity exposed as lizard person truth finally out!",
        "Moon landing filmed in Hollywood basement conspiracy revealed!",
    ]
    
    # Create DataFrames
    df_real = pd.DataFrame({
        "title": ["Real News"] * len(real_texts),
        "text": real_texts,
        "subject": ["politics", "science", "sports", "health"],
        "date": ["2024-01-15"] * len(real_texts),
    })
    
    df_fake = pd.DataFrame({
        "title": ["Fake News"] * len(fake_texts),
        "text": fake_texts,
        "subject": ["conspiracy", "health", "entertainment", "conspiracy"],
        "date": ["2024-01-15"] * len(fake_texts),
    })
    
    # Save to temp files
    real_path = tmp_path / "mini_real.csv"
    fake_path = tmp_path / "mini_fake.csv"
    df_real.to_csv(real_path, index=False)
    df_fake.to_csv(fake_path, index=False)
    
    return {"real_path": real_path, "fake_path": fake_path, "real_df": df_real, "fake_df": df_fake}


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    outdir = tmp_path / "test_outputs"
    outdir.mkdir()
    return outdir


@pytest.fixture
def sample_pipeline():
    """Create a sample trained pipeline for prediction tests."""
    # Small training data
    train_texts = [
        "government announced new policy today",
        "scientists discovered new species",
        "team won championship game",
        "SHOCKING aliens control government",
        "miracle cure doctors hide",
        "celebrity exposed lizard person",
    ]
    train_labels = [0, 0, 0, 1, 1, 1]  # 0=REAL, 1=FAKE
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=100, stop_words="english")),
        ("clf", RandomForestClassifier(n_estimators=10, random_state=42)),
    ])
    pipeline.fit(train_texts, train_labels)
    
    return pipeline


@pytest.fixture
def sample_pipeline_file(tmp_path, sample_pipeline):
    """Save a sample pipeline to a temporary file."""
    pipeline_path = tmp_path / "test_pipeline.joblib"
    joblib.dump(sample_pipeline, pipeline_path)
    return pipeline_path


@pytest.fixture
def sample_separate_artifacts(tmp_path, sample_pipeline):
    """Save vectorizer and model as separate files."""
    vec_path = tmp_path / "test_vectorizer.joblib"
    model_path = tmp_path / "test_model.joblib"
    
    joblib.dump(sample_pipeline.named_steps["tfidf"], vec_path)
    joblib.dump(sample_pipeline.named_steps["clf"], model_path)
    
    return {"vectorizer_path": vec_path, "model_path": model_path}
