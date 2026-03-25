"""
Unit tests for detect_fake_news module.
Tests prediction interface with short texts.
"""

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from detect_fake_news import load_pipeline_or_parts


class TestLoadPipelineOrParts:
    """Tests for load_pipeline_or_parts function."""

    def test_load_pipeline_only(self, tmp_path):
        """Test loading with pipeline path only."""
        # Create a dummy pipeline
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", RandomForestClassifier()),
        ])
        pipeline_path = tmp_path / "pipeline.joblib"
        joblib.dump(pipe, pipeline_path)
        
        loaded_pipe, clf, vec = load_pipeline_or_parts(str(pipeline_path), None, None)
        
        assert loaded_pipe is not None
        assert clf is None
        assert vec is None
        assert isinstance(loaded_pipe, Pipeline)

    def test_load_separate_artifacts(self, tmp_path):
        """Test loading separate vectorizer and model files."""
        # Create and save separate artifacts
        vec = TfidfVectorizer()
        clf = RandomForestClassifier()
        
        # Fit with dummy data so vectorizer has vocabulary
        vec.fit(["test document"])
        
        vec_path = tmp_path / "vectorizer.joblib"
        model_path = tmp_path / "model.joblib"
        joblib.dump(vec, vec_path)
        joblib.dump(clf, model_path)
        
        loaded_pipe, loaded_clf, loaded_vec = load_pipeline_or_parts(
            None, str(model_path), str(vec_path)
        )
        
        assert loaded_pipe is None
        assert loaded_clf is not None
        assert loaded_vec is not None
        assert isinstance(loaded_clf, RandomForestClassifier)
        assert isinstance(loaded_vec, TfidfVectorizer)

    def test_missing_both_raises_error(self):
        """Test error when neither pipeline nor both artifacts provided."""
        with pytest.raises(ValueError, match="Provide --pipeline OR both --model and --vectorizer"):
            load_pipeline_or_parts(None, None, None)

    def test_missing_model_raises_error(self, tmp_path):
        """Test error when only vectorizer provided."""
        vec_path = tmp_path / "vectorizer.joblib"
        joblib.dump(TfidfVectorizer(), vec_path)
        
        with pytest.raises(ValueError, match="Provide --pipeline OR both --model and --vectorizer"):
            load_pipeline_or_parts(None, None, str(vec_path))

    def test_missing_vectorizer_raises_error(self, tmp_path):
        """Test error when only model provided."""
        model_path = tmp_path / "model.joblib"
        joblib.dump(RandomForestClassifier(), model_path)
        
        with pytest.raises(ValueError, match="Provide --pipeline OR both --model and --vectorizer"):
            load_pipeline_or_parts(None, str(model_path), None)


class TestPrediction:
    """Tests for prediction functionality."""

    def test_predict_real_news_with_pipeline(self, sample_pipeline):
        """Test predicting a real news text using pipeline."""
        text = "The government announced new economic policies today."
        
        prob = float(sample_pipeline.predict_proba([text])[0, 1])
        pred = sample_pipeline.predict([text])[0]
        
        assert 0 <= prob <= 1
        assert pred in [0, 1]

    def test_predict_fake_news_with_pipeline(self, sample_pipeline):
        """Test predicting a fake news text using pipeline."""
        text = "SHOCKING conspiracy revealed about government cover up!"
        
        prob = float(sample_pipeline.predict_proba([text])[0, 1])
        pred = sample_pipeline.predict([text])[0]
        
        assert 0 <= prob <= 1
        assert pred in [0, 1]

    def test_predict_with_separate_artifacts(self, sample_separate_artifacts):
        """Test prediction using separate vectorizer and model."""
        from detect_fake_news import load_pipeline_or_parts
        
        vec_path = sample_separate_artifacts["vectorizer_path"]
        model_path = sample_separate_artifacts["model_path"]
        
        _, clf, vec = load_pipeline_or_parts(None, str(model_path), str(vec_path))
        
        text = "Test news article content"
        X = vec.transform([text])
        prob = float(clf.predict_proba(X)[0, 1])
        
        assert 0 <= prob <= 1

    def test_prediction_probability_range(self, sample_pipeline):
        """Test that prediction probabilities are in valid range."""
        texts = [
            "Real government policy announcement",
            "Fake conspiracy theory shock",
            "Normal everyday news article",
        ]
        
        for text in texts:
            probs = sample_pipeline.predict_proba([text])[0]
            assert all(0 <= p <= 1 for p in probs)
            assert abs(sum(probs) - 1.0) < 1e-6  # Probabilities sum to 1

    def test_prediction_consistency(self, sample_pipeline):
        """Test that same input gives same output."""
        text = "Test article for consistency check"
        
        prob1 = sample_pipeline.predict_proba([text])[0, 1]
        prob2 = sample_pipeline.predict_proba([text])[0, 1]
        
        assert prob1 == prob2

    def test_short_text_prediction(self, sample_pipeline):
        """Test prediction with very short text."""
        text = "News"
        
        prob = float(sample_pipeline.predict_proba([text])[0, 1])
        assert 0 <= prob <= 1

    def test_long_text_prediction(self, sample_pipeline):
        """Test prediction with longer text."""
        text = " ".join(["This is a news article."] * 100)
        
        prob = float(sample_pipeline.predict_proba([text])[0, 1])
        assert 0 <= prob <= 1

    def test_empty_text_prediction(self, sample_pipeline):
        """Test prediction with empty text."""
        text = ""
        
        # Should handle empty text without error
        prob = float(sample_pipeline.predict_proba([text])[0, 1])
        assert 0 <= prob <= 1


class TestThresholdBehavior:
    """Tests for prediction threshold behavior."""

    def test_default_threshold(self, sample_pipeline):
        """Test prediction with default threshold (0.5)."""
        text = "Some news article"
        prob = float(sample_pipeline.predict_proba([text])[0, 1])
        
        # With default threshold of 0.5
        label = "FAKE" if prob >= 0.5 else "REAL"
        assert label in ["FAKE", "REAL"]

    def test_custom_threshold(self, sample_pipeline):
        """Test prediction with custom threshold."""
        text = "Some news article"
        prob = float(sample_pipeline.predict_proba([text])[0, 1])
        
        # With custom threshold of 0.3
        threshold = 0.3
        label = "FAKE" if prob >= threshold else "REAL"
        assert label in ["FAKE", "REAL"]

    def test_high_threshold(self, sample_pipeline):
        """Test with high threshold (0.8)."""
        text = "Some news article"
        prob = float(sample_pipeline.predict_proba([text])[0, 1])
        
        threshold = 0.8
        label = "FAKE" if prob >= threshold else "REAL"
        assert label in ["FAKE", "REAL"]

    def test_low_threshold(self, sample_pipeline):
        """Test with low threshold (0.1)."""
        text = "Some news article"
        prob = float(sample_pipeline.predict_proba([text])[0, 1])
        
        threshold = 0.1
        label = "FAKE" if prob >= threshold else "REAL"
        assert label in ["FAKE", "REAL"]


class TestTextCleaningIntegration:
    """Tests for text cleaning integration in prediction."""

    def test_prediction_with_urls(self, sample_pipeline):
        """Test prediction on text with URLs."""
        from text_clean import clean_text
        
        text = "Visit https://example.com for more info about this news"
        cleaned = clean_text(text)
        
        prob = float(sample_pipeline.predict_proba([cleaned])[0, 1])
        assert 0 <= prob <= 1

    def test_prediction_with_emails(self, sample_pipeline):
        """Test prediction on text with emails."""
        from text_clean import clean_text
        
        text = "Contact reporter@news.com for details"
        cleaned = clean_text(text)
        
        prob = float(sample_pipeline.predict_proba([cleaned])[0, 1])
        assert 0 <= prob <= 1

    def test_prediction_with_special_chars(self, sample_pipeline):
        """Test prediction on text with special characters."""
        from text_clean import clean_text
        
        text = "SHOCKING!!! @#$% Conspiracy revealed!!!"
        cleaned = clean_text(text)
        
        prob = float(sample_pipeline.predict_proba([cleaned])[0, 1])
        assert 0 <= prob <= 1


class TestEdgeCases:
    """Edge case tests for detection."""

    def test_single_word_prediction(self, sample_pipeline):
        """Test prediction with single word."""
        text = "Conspiracy"
        
        prob = float(sample_pipeline.predict_proba([text])[0, 1])
        assert 0 <= prob <= 1

    def test_only_punctuation(self, sample_pipeline):
        """Test prediction with only punctuation."""
        text = "!!!???..."
        
        prob = float(sample_pipeline.predict_proba([text])[0, 1])
        assert 0 <= prob <= 1

    def test_only_numbers(self, sample_pipeline):
        """Test prediction with only numbers."""
        text = "12345 67890"
        
        prob = float(sample_pipeline.predict_proba([text])[0, 1])
        assert 0 <= prob <= 1

    def test_unicode_text(self, sample_pipeline):
        """Test prediction with unicode text."""
        from text_clean import clean_text
        
        text = "News about 世界 and café"
        cleaned = clean_text(text)
        
        prob = float(sample_pipeline.predict_proba([cleaned])[0, 1])
        assert 0 <= prob <= 1

    def test_batch_prediction(self, sample_pipeline):
        """Test batch prediction with multiple texts."""
        texts = [
            "Real government announcement",
            "Fake conspiracy theory",
            "Normal sports news",
        ]
        
        probs = sample_pipeline.predict_proba(texts)
        preds = sample_pipeline.predict(texts)
        
        assert probs.shape == (3, 2)
        assert len(preds) == 3
        assert all(0 <= p <= 1 for p in probs.flatten())
