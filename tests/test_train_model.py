"""
Unit tests for train_model module.
"""
import sys
sys.path.insert(0, "src")

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

# Import from train_model
from train_model import (
    ensure_dir,
    read_csv_any,
    pick_text_column,
    plot_confusion_matrix,
    plot_curve,
    LABELS
)


class TestEnsureDir:
    """Tests for ensure_dir function."""

    def test_create_new_directory(self, temp_dir):
        """Test creating a new directory."""
        new_dir = temp_dir / "new_output"
        result = ensure_dir(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_create_nested_directories(self, temp_dir):
        """Test creating nested directories."""
        nested_dir = temp_dir / "level1" / "level2"
        result = ensure_dir(nested_dir)
        assert result.exists()
        assert result.is_dir()

    def test_existing_directory(self, temp_dir):
        """Test ensure_dir on already existing directory."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()
        (existing_dir / "test.txt").write_text("test")
        
        result = ensure_dir(existing_dir)
        assert result.exists()
        assert (existing_dir / "test.txt").exists()


class TestReadCsvAny:
    """Tests for read_csv_any function."""

    def test_read_utf8_csv(self, temp_dir):
        """Test reading UTF-8 encoded CSV."""
        csv_path = temp_dir / "test_utf8.csv"
        csv_path.write_text("col1,col2\nval1,val2\n", encoding="utf-8")
        
        df = read_csv_any(csv_path)
        assert len(df) == 1
        assert df.iloc[0]["col1"] == "val1"

    def test_read_csv_with_nrows(self, temp_dir):
        """Test reading CSV with nrows limit."""
        csv_path = temp_dir / "test_limit.csv"
        content = "col1,col2\n" + "\n".join([f"v{i},v{i}" for i in range(10)])
        csv_path.write_text(content)
        
        df = read_csv_any(csv_path, nrows=3)
        assert len(df) == 3

    def test_read_sample_news(self, sample_news_path):
        """Test reading the sample news CSV file."""
        df = read_csv_any(sample_news_path)
        assert len(df) == 6
        assert "title" in df.columns
        assert "text" in df.columns
        assert "label" in df.columns


class TestPickTextColumn:
    """Tests for pick_text_column function."""

    def test_preferred_column_exists(self):
        """Test when preferred column exists."""
        df = pd.DataFrame({
            "text": ["content1", "content2"],
            "title": ["title1", "title2"]
        })
        result = pick_text_column(df, "text")
        assert result == "text"

    def test_fallback_to_combined_text(self):
        """Test fallback to combined_text column."""
        df = pd.DataFrame({
            "combined_text": ["content1", "content2"],
            "title": ["title1", "title2"]
        })
        result = pick_text_column(df, "non_existent")
        assert result == "combined_text"

    def test_fallback_to_content(self):
        """Test fallback to content column."""
        df = pd.DataFrame({
            "content": ["article1", "article2"],
            "title": ["title1", "title2"]
        })
        result = pick_text_column(df, "missing_column")
        assert result == "content"

    def test_fallback_to_title(self):
        """Test final fallback to title."""
        df = pd.DataFrame({
            "title": ["title1", "title2"],
            "other": ["val1", "val2"]
        })
        result = pick_text_column(df, "not_found")
        assert result == "title"

    def test_no_suitable_column_raises_error(self):
        """Test that error is raised when no suitable column found."""
        df = pd.DataFrame({
            "unknown_col": ["val1", "val2"],
            "another_col": ["val3", "val4"]
        })
        with pytest.raises(ValueError):
            pick_text_column(df, "missing")


class TestPlotUtilities:
    """Tests for plotting utilities."""

    def test_plot_confusion_matrix(self, temp_dir):
        """Test confusion matrix plotting."""
        import numpy as np
        cm = np.array([[50, 2], [3, 45]])
        out_path = temp_dir / "confusion_matrix.png"
        
        plot_confusion_matrix(cm, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_plot_curve(self, temp_dir):
        """Test generic curve plotting."""
        import numpy as np
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        out_path = temp_dir / "curve.png"
        
        plot_curve(x, y, out_path, "Test Curve", "X Label", "Y Label")
        assert out_path.exists()
        assert out_path.stat().st_size > 0


class TestTrainingPipeline:
    """Tests for the full training pipeline."""

    def test_pipeline_creation(self):
        """Test creating the model pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        
        pipe = Pipeline(steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=100)),
            ("clf", RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        assert isinstance(pipe, Pipeline)
        assert "tfidf" in pipe.named_steps
        assert "clf" in pipe.named_steps

    def test_pipeline_training(self, small_dataset):
        """Test that pipeline can be trained on small dataset."""
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = small_dataset
        
        pipe = Pipeline(steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=100)),
            ("clf", RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        pipe.fit(X, y)
        
        # Verify it can predict
        predictions = pipe.predict(X)
        assert len(predictions) == len(y)
        assert all(pred in [0, 1] for pred in predictions)

    def test_pipeline_persistence(self, temp_dir, small_dataset):
        """Test saving and loading pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = small_dataset
        
        pipe = Pipeline(steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=100)),
            ("clf", RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        pipe.fit(X, y)
        
        # Save pipeline
        model_path = temp_dir / "model.joblib"
        vec_path = temp_dir / "vectorizer.joblib"
        pipe_path = temp_dir / "pipeline.joblib"
        
        joblib.dump(pipe, pipe_path)
        joblib.dump(pipe.named_steps["clf"], model_path)
        joblib.dump(pipe.named_steps["tfidf"], vec_path)
        
        # Verify files exist
        assert pipe_path.exists()
        assert model_path.exists()
        assert vec_path.exists()
        
        # Load and verify consistency
        loaded_pipe = joblib.load(pipe_path)
        original_pred = pipe.predict(X)
        loaded_pred = loaded_pipe.predict(X)
        np.testing.assert_array_equal(original_pred, loaded_pred)

    def test_full_training_workflow(self, temp_dir, sample_news_df):
        """Test full training workflow with sample data."""
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        # Prepare data
        df_real = sample_news_df[sample_news_df["label"] == 0].copy()
        df_fake = sample_news_df[sample_news_df["label"] == 1].copy()
        
        # Build combined text
        for df in (df_real, df_fake):
            title = df["title"].fillna("")
            txt = df["text"].fillna("")
            df["combined_text"] = (title + " " + txt).str.strip()
        
        X = pd.concat([df_real["combined_text"], df_fake["combined_text"]], ignore_index=True)
        y = np.array([0] * len(df_real) + [1] * len(df_fake))
        
        # Create and train pipeline
        pipe = Pipeline(steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=1000)),
            ("clf", RandomForestClassifier(n_estimators=50, random_state=42))
        ])
        pipe.fit(X, y)
        
        # Evaluate
        y_pred = pipe.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # Should have decent accuracy on this simple dataset
        assert accuracy >= 0.5  # At least better than random
        
        # Save artifacts
        outdir = temp_dir / "outputs"
        outdir.mkdir()
        
        joblib.dump(pipe, outdir / "pipeline.joblib")
        joblib.dump(pipe.named_steps["tfidf"], outdir / "vectorizer.joblib")
        joblib.dump(pipe.named_steps["clf"], outdir / "model.joblib")
        
        # Verify artifacts
        assert (outdir / "pipeline.joblib").exists()
        assert (outdir / "vectorizer.joblib").exists()
        assert (outdir / "model.joblib").exists()
        
        # Predict probabilities
        y_prob = pipe.predict_proba(X)[:, 1]
        assert len(y_prob) == len(y)
        assert all(0 <= prob <= 1 for prob in y_prob)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
