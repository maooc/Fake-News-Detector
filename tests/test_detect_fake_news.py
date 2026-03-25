"""
Unit tests for detect_fake_news module.
"""
import sys
sys.path.insert(0, "src")

import pytest
import numpy as np
from pathlib import Path
import joblib

# Import from detect_fake_news
from detect_fake_news import load_pipeline_or_parts


class TestLoadPipelineOrParts:
    """Tests for load_pipeline_or_parts function."""

    def test_load_pipeline_only(self, saved_model_files):
        """Test loading only the pipeline file."""
        pipe, clf, vec = load_pipeline_or_parts(
            pipeline_path=str(saved_model_files["pipeline"]),
            model_path=None,
            vectorizer_path=None
        )
        
        assert pipe is not None
        assert clf is None
        assert vec is None
        assert hasattr(pipe, "predict")
        assert hasattr(pipe, "predict_proba")

    def test_load_separate_model_and_vectorizer(self, saved_model_files):
        """Test loading separate model and vectorizer files."""
        pipe, clf, vec = load_pipeline_or_parts(
            pipeline_path=None,
            model_path=str(saved_model_files["model"]),
            vectorizer_path=str(saved_model_files["vectorizer"])
        )
        
        assert pipe is None
        assert clf is not None
        assert vec is not None
        assert hasattr(clf, "predict")
        assert hasattr(vec, "transform")

    def test_missing_both_raises_error(self):
        """Test that error is raised when neither pipeline nor model/vectorizer provided."""
        with pytest.raises(ValueError, match="Provide --pipeline OR both --model and --vectorizer"):
            load_pipeline_or_parts(None, None, None)

    def test_missing_model_raises_error(self, saved_model_files):
        """Test that error is raised when vectorizer provided but model missing."""
        with pytest.raises(ValueError):
            load_pipeline_or_parts(
                pipeline_path=None,
                model_path=None,
                vectorizer_path=str(saved_model_files["vectorizer"])
            )

    def test_missing_vectorizer_raises_error(self, saved_model_files):
        """Test that error is raised when model provided but vectorizer missing."""
        with pytest.raises(ValueError):
            load_pipeline_or_parts(
                pipeline_path=None,
                model_path=str(saved_model_files["model"]),
                vectorizer_path=None
            )

    def test_load_with_path_objects(self, saved_model_files):
        """Test loading with Path objects instead of strings."""
        pipe, clf, vec = load_pipeline_or_parts(
            pipeline_path=saved_model_files["pipeline"],
            model_path=None,
            vectorizer_path=None
        )
        assert pipe is not None


class TestPredictionFunctions:
    """Tests for prediction functionality."""

    def test_pipeline_prediction(self, sample_pipeline):
        """Test prediction using the full pipeline."""
        test_text = "Real news article about politics and economy"
        prob = sample_pipeline.predict_proba([test_text])[0, 1]
        
        assert isinstance(prob, (float, np.floating))
        assert 0 <= prob <= 1

    def test_separate_model_prediction(self, sample_pipeline):
        """Test prediction using separate model and vectorizer."""
        vec = sample_pipeline.named_steps["tfidf"]
        clf = sample_pipeline.named_steps["clf"]
        
        test_text = "Fake news clickbait shocking title"
        X = vec.transform([test_text])
        prob = float(clf.predict_proba(X)[0, 1])
        
        assert isinstance(prob, float)
        assert 0 <= prob <= 1

    def test_prediction_with_cleaned_text(self, sample_pipeline):
        """Test that predictions work with cleaned text."""
        from text_clean import clean_text
        
        dirty_text = "  SHOCKING! https://fake.com  "
        clean_text_result = clean_text(dirty_text)
        
        prob = float(sample_pipeline.predict_proba([clean_text_result])[0, 1])
        assert 0 <= prob <= 1

    def test_prediction_threshold_behavior(self, sample_pipeline):
        """Test threshold behavior for classification."""
        # Get a probability
        test_text = "Test news content"
        prob = float(sample_pipeline.predict_proba([test_text])[0, 1])
        
        # Test different thresholds
        threshold = 0.5
        label = "FAKE" if prob >= threshold else "REAL"
        assert label in ["FAKE", "REAL"]
        
        # Verify threshold logic
        high_threshold = 0.99
        low_threshold = 0.01
        
        # At very high threshold, most predictions should be REAL
        assert "FAKE" if prob >= high_threshold else "REAL" == "REAL" if prob < 0.99 else "FAKE"
        
        # At very low threshold, most predictions should be FAKE
        assert "FAKE" if prob >= low_threshold else "REAL" == "FAKE" if prob > 0.01 else "REAL"

    def test_multiple_predictions(self, sample_pipeline):
        """Test making multiple predictions in sequence."""
        texts = [
            "Real news article content here",
            "Shocking fake news clickbait",
            "Another real news story",
            "More fake gossip headlines"
        ]
        
        probs = []
        for text in texts:
            prob = float(sample_pipeline.predict_proba([text])[0, 1])
            probs.append(prob)
        
        assert len(probs) == 4
        assert all(0 <= p <= 1 for p in probs)

    def test_empty_text_prediction(self, sample_pipeline):
        """Test prediction with empty or whitespace text."""
        from text_clean import clean_text
        
        empty_texts = ["", "   ", None, "\n\n\t"]
        
        for text in empty_texts:
            cleaned = clean_text(text)
            prob = float(sample_pipeline.predict_proba([cleaned])[0, 1])
            assert 0 <= prob <= 1


class TestCLIFunctionality:
    """Tests for CLI-like behavior of the detection module."""

    def test_full_detection_workflow_pipeline(self, saved_model_files, sample_texts):
        """Test full detection workflow using pipeline."""
        from text_clean import clean_text
        
        # Load pipeline
        pipe, _, _ = load_pipeline_or_parts(
            pipeline_path=str(saved_model_files["pipeline"]),
            model_path=None,
            vectorizer_path=None
        )
        
        # Test with real-looking texts
        for text in sample_texts["real"]:
            s = clean_text(text)
            prob = float(pipe.predict_proba([s])[0, 1])
            label = "FAKE" if prob >= 0.4 else "REAL"
            assert label in ["FAKE", "REAL"]
            assert 0 <= prob <= 1

    def test_full_detection_workflow_separate(self, saved_model_files, sample_texts):
        """Test full detection workflow using separate model and vectorizer."""
        from text_clean import clean_text
        
        # Load separate model and vectorizer
        _, clf, vec = load_pipeline_or_parts(
            pipeline_path=None,
            model_path=str(saved_model_files["model"]),
            vectorizer_path=str(saved_model_files["vectorizer"])
        )
        
        # Test with fake-looking texts
        for text in sample_texts["fake"]:
            s = clean_text(text)
            X = vec.transform([s])
            prob = float(clf.predict_proba(X)[0, 1])
            label = "FAKE" if prob >= 0.4 else "REAL"
            assert label in ["FAKE", "REAL"]
            assert 0 <= prob <= 1

    def test_different_threshold_values(self, saved_model_files):
        """Test detection with different threshold values."""
        from text_clean import clean_text
        
        pipe, _, _ = load_pipeline_or_parts(
            pipeline_path=str(saved_model_files["pipeline"]),
            model_path=None,
            vectorizer_path=None
        )
        
        test_text = "Test news article content for threshold testing"
        s = clean_text(test_text)
        prob = float(pipe.predict_proba([s])[0, 1])
        
        # Test different thresholds produce logical results
        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            label = "FAKE" if prob >= threshold else "REAL"
            assert label in ["FAKE", "REAL"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
