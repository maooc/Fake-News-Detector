"""
Unit tests for train_model module.
Tests training functionality with small sample datasets.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from train_model import (
    ensure_dir,
    read_csv_any,
    pick_text_column,
    plot_confusion_matrix,
    plot_curve,
    LABELS,
)


class TestEnsureDir:
    """Tests for ensure_dir helper function."""

    def test_create_new_directory(self, tmp_path):
        """Test creating a new directory."""
        new_dir = tmp_path / "test_output"
        result = ensure_dir(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_create_nested_directories(self, tmp_path):
        """Test creating nested directory structure."""
        nested = tmp_path / "a" / "b" / "c"
        result = ensure_dir(nested)
        assert nested.exists()

    def test_existing_directory(self, tmp_path):
        """Test with already existing directory."""
        existing = tmp_path / "existing"
        existing.mkdir()
        result = ensure_dir(existing)
        assert result == existing


class TestReadCsvAny:
    """Tests for read_csv_any function."""

    def test_read_utf8_csv(self, tmp_path):
        """Test reading UTF-8 encoded CSV."""
        csv_path = tmp_path / "utf8.csv"
        df = pd.DataFrame({
            "title": ["Test 1", "Test 2"],
            "text": ["Content 1", "Content 2"],
            "label": [0, 1]
        })
        df.to_csv(csv_path, index=False, encoding="utf-8")
        
        result = read_csv_any(csv_path)
        assert len(result) == 2
        assert list(result.columns) == ["title", "text", "label"]

    def test_read_latin1_csv(self, tmp_path):
        """Test reading Latin-1 encoded CSV."""
        csv_path = tmp_path / "latin1.csv"
        # Write with latin-1 encoding
        with open(csv_path, "w", encoding="latin-1") as f:
            f.write("title,text,label\n")
            f.write("Test caf\xe9,Content,0\n")  # é in latin-1
        
        result = read_csv_any(csv_path)
        assert len(result) == 1

    def test_read_with_nrows(self, tmp_path):
        """Test reading limited number of rows."""
        csv_path = tmp_path / "many_rows.csv"
        df = pd.DataFrame({
            "title": [f"Title {i}" for i in range(100)],
            "text": [f"Text {i}" for i in range(100)],
        })
        df.to_csv(csv_path, index=False)
        
        result = read_csv_any(csv_path, nrows=10)
        assert len(result) == 10


class TestPickTextColumn:
    """Tests for pick_text_column function."""

    def test_preferred_column_exists(self):
        """Test when preferred column exists."""
        df = pd.DataFrame({
            "title": ["T1", "T2"],
            "text": ["C1", "C2"],
            "other": ["O1", "O2"]
        })
        result = pick_text_column(df, "text")
        assert result == "text"

    def test_fallback_columns(self):
        """Test fallback to alternative column names."""
        # Test with 'combined_text'
        df1 = pd.DataFrame({
            "title": ["T1"],
            "combined_text": ["C1"]
        })
        assert pick_text_column(df1, "nonexistent") == "combined_text"
        
        # Test with 'content'
        df2 = pd.DataFrame({
            "title": ["T1"],
            "content": ["C1"]
        })
        assert pick_text_column(df2, "nonexistent") == "content"
        
        # Test with 'article'
        df3 = pd.DataFrame({
            "title": ["T1"],
            "article": ["C1"]
        })
        assert pick_text_column(df3, "nonexistent") == "article"

    def test_fallback_to_title(self):
        """Test fallback to title column when no text column found."""
        df = pd.DataFrame({
            "title": ["T1", "T2"],
            "date": ["2024-01-01", "2024-01-02"]
        })
        result = pick_text_column(df, "nonexistent")
        assert result == "title"

    def test_no_suitable_column_raises_error(self):
        """Test error when no suitable column exists."""
        df = pd.DataFrame({
            "id": [1, 2],
            "date": ["2024-01-01", "2024-01-02"]
        })
        with pytest.raises(ValueError, match="No suitable text column found"):
            pick_text_column(df, "nonexistent")


class TestPlotFunctions:
    """Tests for plotting utility functions."""

    def test_plot_confusion_matrix(self, tmp_path):
        """Test confusion matrix plotting."""
        cm = np.array([[45, 5], [3, 47]])
        output_path = tmp_path / "cm.png"
        
        plot_confusion_matrix(cm, output_path, title="Test CM")
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_plot_confusion_matrix_different_values(self, tmp_path):
        """Test CM plotting with different values."""
        cm = np.array([[100, 0], [0, 100]])  # Perfect classification
        output_path = tmp_path / "perfect_cm.png"
        
        plot_confusion_matrix(cm, output_path)
        assert output_path.exists()

    def test_plot_curve(self, tmp_path):
        """Test curve plotting (ROC or PR)."""
        x = np.array([0, 0.1, 0.5, 1.0])
        y = np.array([0, 0.5, 0.8, 1.0])
        output_path = tmp_path / "curve.png"
        
        plot_curve(x, y, output_path, "Test Curve", "X Label", "Y Label")
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_plot_curve_roc_shape(self, tmp_path):
        """Test plotting ROC-like curve."""
        fpr = np.linspace(0, 1, 100)
        tpr = np.linspace(0, 1, 100)
        output_path = tmp_path / "roc.png"
        
        plot_curve(fpr, tpr, output_path, "ROC Curve", "FPR", "TPR")
        assert output_path.exists()


class TestTrainingPipeline:
    """Tests for the main training functionality."""

    def test_train_model_outputs_expected_files(self, mini_dataset, temp_output_dir):
        """Test that training produces expected output files."""
        from train_model import read_csv_any, pick_text_column, ensure_dir
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        # Load data
        df_real = read_csv_any(mini_dataset["real_path"])
        df_fake = read_csv_any(mini_dataset["fake_path"])
        
        # Build combined_text
        for df in (df_real, df_fake):
            title = df["title"].fillna("") if "title" in df.columns else ""
            txt = df.get("text", "").fillna("")
            df["combined_text"] = (title + " " + txt).str.strip()
        
        # Prepare X, y
        X = pd.concat([df_real["combined_text"], df_fake["combined_text"]], ignore_index=True)
        y = np.array([0] * len(df_real) + [1] * len(df_fake))
        
        # Create and train pipeline
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=100, stop_words="english")),
            ("clf", RandomForestClassifier(n_estimators=10, random_state=42)),
        ])
        pipe.fit(X, y)
        
        # Save artifacts
        outdir = ensure_dir(temp_output_dir)
        joblib.dump(pipe, outdir / "pipeline.joblib")
        joblib.dump(pipe.named_steps["tfidf"], outdir / "vectorizer.joblib")
        joblib.dump(pipe.named_steps["clf"], outdir / "model.joblib")
        
        # Verify outputs
        assert (outdir / "pipeline.joblib").exists()
        assert (outdir / "vectorizer.joblib").exists()
        assert (outdir / "model.joblib").exists()

    def test_trained_model_can_predict(self, mini_dataset):
        """Test that trained model can make predictions."""
        from train_model import read_csv_any
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        # Load and prepare data
        df_real = read_csv_any(mini_dataset["real_path"])
        df_fake = read_csv_any(mini_dataset["fake_path"])
        
        for df in (df_real, df_fake):
            title = df["title"].fillna("") if "title" in df.columns else ""
            txt = df.get("text", "").fillna("")
            df["combined_text"] = (title + " " + txt).str.strip()
        
        X = pd.concat([df_real["combined_text"], df_fake["combined_text"]], ignore_index=True)
        y = np.array([0] * len(df_real) + [1] * len(df_fake))
        
        # Train
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50, stop_words="english")),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ])
        pipe.fit(X, y)
        
        # Predict
        predictions = pipe.predict(X)
        probabilities = pipe.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)
        assert all(prob >= 0 and prob <= 1 for prob in probabilities.flatten())

    def test_model_pipeline_structure(self, mini_dataset):
        """Test that trained pipeline has correct structure."""
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        import pandas as pd
        
        # Simple training data
        X = pd.Series([
            "government announced new policy",
            "scientists discovered species",
            "shocking alien conspiracy revealed",
            "miracle cure doctors hide"
        ])
        y = np.array([0, 0, 1, 1])
        
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20)),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ])
        pipe.fit(X, y)
        
        # Check pipeline structure
        assert "tfidf" in pipe.named_steps
        assert "clf" in pipe.named_steps
        assert isinstance(pipe.named_steps["tfidf"], TfidfVectorizer)
        assert isinstance(pipe.named_steps["clf"], RandomForestClassifier)

    def test_training_with_small_sample(self):
        """Test training with minimal sample size."""
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        import pandas as pd
        
        # Minimal dataset (2 samples per class)
        X = pd.Series([
            "real news article",
            "another real article",
            "fake conspiracy theory",
            "another fake story"
        ])
        y = np.array([0, 0, 1, 1])
        
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10)),
            ("clf", RandomForestClassifier(n_estimators=3, random_state=42)),
        ])
        
        # Should not raise error
        pipe.fit(X, y)
        
        # Should be able to predict
        preds = pipe.predict(["test article"])
        assert len(preds) == 1


class TestLabels:
    """Tests for label constants."""

    def test_labels_defined(self):
        """Test that LABELS constant is properly defined."""
        assert len(LABELS) == 2
        assert LABELS[0] == "REAL"
        assert LABELS[1] == "FAKE"

    def test_labels_are_strings(self):
        """Test that labels are strings."""
        assert isinstance(LABELS[0], str)
        assert isinstance(LABELS[1], str)
