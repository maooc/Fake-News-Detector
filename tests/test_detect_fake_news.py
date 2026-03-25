import sys
import pytest
import joblib
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from detect_fake_news import load_pipeline_or_parts
from text_clean import clean_text


class TestLoadPipelineOrParts:
    def test_load_pipeline_only(self, temp_output_dir):
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ])
        texts = ["real news about politics", "fake shocking news"]
        labels = [0, 1]
        pipe.fit(texts, labels)

        pipeline_path = temp_output_dir / "pipeline.joblib"
        joblib.dump(pipe, pipeline_path)

        loaded_pipe, clf, vec = load_pipeline_or_parts(
            str(pipeline_path), None, None
        )
        assert loaded_pipe is not None
        assert clf is None
        assert vec is None

    def test_load_separate_parts(self, temp_output_dir):
        vec = TfidfVectorizer()
        clf = RandomForestClassifier(n_estimators=5, random_state=42)

        texts = ["real news", "fake news"]
        X = vec.fit_transform(texts)
        clf.fit(X, [0, 1])

        model_path = temp_output_dir / "model.joblib"
        vec_path = temp_output_dir / "vectorizer.joblib"
        joblib.dump(clf, model_path)
        joblib.dump(vec, vec_path)

        pipe, loaded_clf, loaded_vec = load_pipeline_or_parts(
            None, str(model_path), str(vec_path)
        )
        assert pipe is None
        assert loaded_clf is not None
        assert loaded_vec is not None

    def test_missing_both_model_and_vectorizer(self, temp_output_dir):
        with pytest.raises(ValueError, match="Provide --pipeline OR both"):
            load_pipeline_or_parts(None, None, None)

    def test_missing_vectorizer(self, temp_output_dir):
        model_path = temp_output_dir / "model.joblib"
        clf = RandomForestClassifier(n_estimators=5)
        joblib.dump(clf, model_path)

        with pytest.raises(ValueError, match="Provide --pipeline OR both"):
            load_pipeline_or_parts(None, str(model_path), None)


class TestPredictionInterface:
    @pytest.fixture
    def trained_pipeline(self):
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
            )),
            ("clf", RandomForestClassifier(
                n_estimators=10,
                random_state=42,
                n_jobs=1,
            )),
        ])

        texts = [
            "Scientists discover new planet in solar system",
            "Government announces new economic policy",
            "SHOCKING: Aliens found in backyard! You won't believe!",
            "MIRACLE cure doctors don't want you to know about!",
            "Research shows climate change effects on agriculture",
            "Election results show clear winner in local race",
            "CELEBRITY secret revealed! Click here now!",
            "BREAKING: Government hiding truth about aliens!",
        ]
        labels = [0, 0, 1, 1, 0, 0, 1, 1]

        pipe.fit(texts, labels)
        return pipe

    def test_predict_returns_probability(self, trained_pipeline):
        text = "New scientific discovery announced by researchers"
        cleaned = clean_text(text)
        prob = float(trained_pipeline.predict_proba([cleaned])[0, 1])

        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_predict_returns_valid_label(self, trained_pipeline):
        text = "SHOCKING news that will blow your mind"
        cleaned = clean_text(text)
        pred = trained_pipeline.predict([cleaned])[0]

        assert pred in [0, 1]

    def test_predict_real_news_style(self, trained_pipeline):
        real_texts = [
            "The government announced a new policy today",
            "Scientists published research in a peer-reviewed journal",
            "Local election results were certified by officials",
        ]

        for text in real_texts:
            cleaned = clean_text(text)
            prob = float(trained_pipeline.predict_proba([cleaned])[0, 1])
            assert isinstance(prob, float)

    def test_predict_fake_news_style(self, trained_pipeline):
        fake_texts = [
            "SHOCKING truth they don't want you to know!",
            "You won't BELIEVE what happened next! Click here!",
            "Doctors HATE this one simple trick!",
        ]

        for text in fake_texts:
            cleaned = clean_text(text)
            prob = float(trained_pipeline.predict_proba([cleaned])[0, 1])
            assert isinstance(prob, float)

    def test_threshold_logic(self, trained_pipeline):
        text = "Some news article text"
        cleaned = clean_text(text)
        prob = float(trained_pipeline.predict_proba([cleaned])[0, 1])

        threshold = 0.5
        label = "FAKE" if prob >= threshold else "REAL"
        assert label in ["REAL", "FAKE"]

    def test_custom_threshold(self, trained_pipeline):
        text = "Breaking news about important events"
        cleaned = clean_text(text)
        prob = float(trained_pipeline.predict_proba([cleaned])[0, 1])

        low_threshold = 0.1
        label_low = "FAKE" if prob >= low_threshold else "REAL"

        high_threshold = 0.9
        label_high = "FAKE" if prob >= high_threshold else "REAL"

        assert label_low in ["REAL", "FAKE"]
        assert label_high in ["REAL", "FAKE"]


class TestTextCleaningIntegration:
    @pytest.fixture
    def simple_pipeline(self):
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(min_df=1)),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ])
        texts = ["clean text here", "dirty text there"]
        labels = [0, 1]
        pipe.fit(texts, labels)
        return pipe

    def test_prediction_with_url_removal(self, simple_pipeline):
        text_with_url = "Check https://example.com for clean text here"
        cleaned = clean_text(text_with_url)
        prob = float(simple_pipeline.predict_proba([cleaned])[0, 1])
        assert isinstance(prob, float)

    def test_prediction_with_email_removal(self, simple_pipeline):
        text_with_email = "Contact test@email.com for clean text here"
        cleaned = clean_text(text_with_email)
        prob = float(simple_pipeline.predict_proba([cleaned])[0, 1])
        assert isinstance(prob, float)

    def test_prediction_with_special_chars(self, simple_pipeline):
        text_with_special = "Clean text here!!! With symbols @#$%"
        cleaned = clean_text(text_with_special)
        prob = float(simple_pipeline.predict_proba([cleaned])[0, 1])
        assert isinstance(prob, float)


class TestEndToEndPrediction:
    @pytest.fixture
    def full_pipeline(self, temp_output_dir):
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
            )),
            ("clf", RandomForestClassifier(
                n_estimators=10,
                random_state=42,
                n_jobs=1,
            )),
        ])

        training_texts = [
            "Scientists publish new research findings",
            "Government releases economic data",
            "Local news reports on community events",
            "SHOCKING revelation about celebrities!",
            "Doctors hate this one weird trick!",
            "You won't believe what happened next!",
        ]
        labels = [0, 0, 0, 1, 1, 1]

        pipe.fit(training_texts, labels)

        pipeline_path = temp_output_dir / "test_pipeline.joblib"
        joblib.dump(pipe, pipeline_path)

        return pipeline_path

    def test_load_and_predict(self, full_pipeline):
        loaded_pipe, _, _ = load_pipeline_or_parts(str(full_pipeline), None, None)

        test_text = "New research published in scientific journal"
        cleaned = clean_text(test_text)
        prob = float(loaded_pipe.predict_proba([cleaned])[0, 1])

        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_multiple_predictions(self, full_pipeline):
        loaded_pipe, _, _ = load_pipeline_or_parts(str(full_pipeline), None, None)

        test_texts = [
            "Government announces policy changes",
            "SHOCKING news you must read!",
            "Research shows interesting results",
        ]

        for text in test_texts:
            cleaned = clean_text(text)
            prob = float(loaded_pipe.predict_proba([cleaned])[0, 1])
            assert isinstance(prob, float)
