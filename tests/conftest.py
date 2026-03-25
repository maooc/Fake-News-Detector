"""
Pytest fixtures for fake news detector tests.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


@pytest.fixture
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_fixtures_dir(project_root):
    """Return test fixtures directory."""
    return project_root / "tests" / "fixtures"


@pytest.fixture
def sample_news_path(test_fixtures_dir):
    """Return path to sample news CSV."""
    return test_fixtures_dir / "sample_news.csv"


@pytest.fixture
def test_data_path(test_fixtures_dir):
    """Return path to test data JSON."""
    return test_fixtures_dir / "test_data.json"


@pytest.fixture
def sample_news_df(sample_news_path):
    """Return sample news DataFrame."""
    return pd.read_csv(sample_news_path)


@pytest.fixture
def test_data(test_data_path):
    """Return test data from JSON file."""
    with open(test_data_path, "r") as f:
        return json.load(f)


@pytest.fixture
def temp_dir(tmp_path):
    """Return a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def sample_texts():
    """Return sample text data for testing."""
    return {
        "real": [
            "The economy shows positive growth in the latest quarter.",
            "Scientists discover new renewable energy source.",
            "Local community comes together for charity event."
        ],
        "fake": [
            "SHOCKING: Government hides alien contact for 50 years!",
            "Miracle weight loss pill burns fat while you sleep!",
            "Celebrity secret life exposed - you won't believe what happened!"
        ],
        "dirty": [
            "  Check out this link: https://example.com/spam  ",
            "Contact me at spam@email.com for more info!",
            "Special chars: ñáéíóú € £ ¥",
            "Multiple    spaces    and    newlines\n\n\tshould be cleaned"
        ]
    }


@pytest.fixture
def small_dataset():
    """Return small balanced dataset for training tests."""
    X = pd.Series([
        "Real news about politics and economy",
        "Scientific breakthrough in medical research",
        "Sports team wins championship game",
        "Shocking celebrity gossip exposed now!",
        "Miracle cure discovered by local doctor",
        "Conspiracy theory about secret societies"
    ])
    y = np.array([0, 0, 0, 1, 1, 1])  # 0=REAL, 1=FAKE
    return X, y


@pytest.fixture
def sample_pipeline():
    """Create a simple trained pipeline for testing."""
    X = [
        "Real news article content",
        "Another real news story",
        "Fake news clickbait title",
        "Another fake news article"
    ]
    y = [0, 0, 1, 1]
    
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=100)),
        ("clf", RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    pipe.fit(X, y)
    return pipe


@pytest.fixture
def saved_model_files(temp_dir, sample_pipeline):
    """Save pipeline and separate model files for testing."""
    pipeline_path = temp_dir / "pipeline.joblib"
    model_path = temp_dir / "model.joblib"
    vectorizer_path = temp_dir / "vectorizer.joblib"
    
    joblib.dump(sample_pipeline, pipeline_path)
    joblib.dump(sample_pipeline.named_steps["clf"], model_path)
    joblib.dump(sample_pipeline.named_steps["tfidf"], vectorizer_path)
    
    return {
        "pipeline": pipeline_path,
        "model": model_path,
        "vectorizer": vectorizer_path
    }
