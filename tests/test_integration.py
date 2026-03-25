"""
Integration and end-to-end tests for the fake news detector.
Tests the complete workflow: data loading -> training -> saving -> loading -> prediction.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


class TestEndToEndWorkflow:
    """End-to-end tests for the complete detection workflow."""

    def test_full_pipeline_workflow(self, tmp_path):
        """
        Complete end-to-end test:
        1. Create synthetic data
        2. Train model
        3. Save artifacts
        4. Load artifacts
        5. Make predictions
        """
        from train_model import read_csv_any, ensure_dir
        from detect_fake_news import load_pipeline_or_parts
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        
        # Step 1: Create synthetic training data
        outdir = ensure_dir(tmp_path / "test_outputs")
        
        real_texts = [
            "The government announced new economic policies today in Washington DC.",
            "Scientists discovered a new species of frog in the Amazon rainforest.",
            "The local basketball team won the championship after overtime.",
            "New hospital opened downtown to serve the growing community needs.",
            "Education reform bill passed by lawmakers with bipartisan support.",
        ]
        
        fake_texts = [
            "SHOCKING secret documents reveal aliens control the government!",
            "Miracle cure doctors don't want you to know about this secret!",
            "Celebrity exposed as lizard person the truth is finally out!",
            "Moon landing was filmed in Hollywood basement conspiracy revealed!",
            "Free energy device suppressed by oil companies inventor assassinated!",
        ]
        
        # Create DataFrames
        df_real = pd.DataFrame({
            "title": ["Real News"] * len(real_texts),
            "text": real_texts,
            "subject": ["politics", "science", "sports", "health", "politics"],
            "date": ["2024-01-15"] * len(real_texts),
        })
        
        df_fake = pd.DataFrame({
            "title": ["Fake News"] * len(fake_texts),
            "text": fake_texts,
            "subject": ["conspiracy", "health", "entertainment", "conspiracy", "technology"],
            "date": ["2024-01-15"] * len(fake_texts),
        })
        
        # Save to CSV
        real_path = tmp_path / "synthetic_real.csv"
        fake_path = tmp_path / "synthetic_fake.csv"
        df_real.to_csv(real_path, index=False)
        df_fake.to_csv(fake_path, index=False)
        
        # Step 2: Load and prepare data (as train_model does)
        df_real_loaded = read_csv_any(real_path)
        df_fake_loaded = read_csv_any(fake_path)
        
        for df in (df_real_loaded, df_fake_loaded):
            title = df["title"].fillna("") if "title" in df.columns else ""
            txt = df.get("text", "").fillna("")
            df["combined_text"] = (title + " " + txt).str.strip()
        
        X = pd.concat([df_real_loaded["combined_text"], df_fake_loaded["combined_text"]], ignore_index=True)
        y = np.array([0] * len(df_real_loaded) + [1] * len(df_fake_loaded))
        
        # Step 3: Train pipeline
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=100, stop_words="english", ngram_range=(1, 2))),
            ("clf", RandomForestClassifier(n_estimators=20, random_state=42)),
        ])
        pipe.fit(X, y)
        
        # Step 4: Save artifacts
        pipeline_path = outdir / "pipeline.joblib"
        vectorizer_path = outdir / "vectorizer.joblib"
        model_path = outdir / "model.joblib"
        
        joblib.dump(pipe, pipeline_path)
        joblib.dump(pipe.named_steps["tfidf"], vectorizer_path)
        joblib.dump(pipe.named_steps["clf"], model_path)
        
        # Verify files exist
        assert pipeline_path.exists()
        assert vectorizer_path.exists()
        assert model_path.exists()
        
        # Step 5: Load and predict using pipeline
        loaded_pipe, _, _ = load_pipeline_or_parts(str(pipeline_path), None, None)
        
        test_texts = [
            "The government announced new policy today",  # Should be REAL
            "SHOCKING conspiracy revealed about aliens",  # Should be FAKE
        ]
        
        for text in test_texts:
            prob = float(loaded_pipe.predict_proba([text])[0, 1])
            assert 0 <= prob <= 1
            pred = loaded_pipe.predict([text])[0]
            assert pred in [0, 1]

    def test_train_save_load_predict_with_separate_artifacts(self, tmp_path):
        """
        Test workflow using separate vectorizer and model files.
        """
        from train_model import ensure_dir
        from detect_fake_news import load_pipeline_or_parts
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        
        outdir = ensure_dir(tmp_path / "artifacts")
        
        # Create minimal training data
        train_texts = [
            "government policy announcement real news",
            "scientists discover new species research",
            "team wins championship sports victory",
            "SHOCKING conspiracy theory fake news",
            "miracle cure secret doctors hide",
            "celebrity scandal exposed truth revealed",
        ]
        train_labels = [0, 0, 0, 1, 1, 1]  # 0=REAL, 1=FAKE
        
        # Train
        vec = TfidfVectorizer(max_features=50)
        X = vec.fit_transform(train_texts)
        
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, train_labels)
        
        # Save separately
        vec_path = outdir / "vectorizer.joblib"
        model_path = outdir / "model.joblib"
        joblib.dump(vec, vec_path)
        joblib.dump(clf, model_path)
        
        # Load separately and predict
        _, loaded_clf, loaded_vec = load_pipeline_or_parts(None, str(model_path), str(vec_path))
        
        test_text = "government announced new policy today"
        X_test = loaded_vec.transform([test_text])
        prob = float(loaded_clf.predict_proba(X_test)[0, 1])
        
        assert 0 <= prob <= 1

    def test_metrics_json_workflow(self, tmp_path):
        """
        Test saving and loading metrics JSON.
        """
        from utils import save_json, load_json
        
        # Create metrics as train_model would
        metrics = {
            "accuracy": 0.95,
            "roc_auc": 0.97,
            "avg_precision": 0.94,
            "cv_f1_macro_mean": 0.93,
            "cv_f1_macro_std": 0.02,
            "report": {
                "REAL": {"precision": 0.96, "recall": 0.94, "f1-score": 0.95},
                "FAKE": {"precision": 0.94, "recall": 0.96, "f1-score": 0.95},
            }
        }
        
        metrics_path = tmp_path / "metrics.json"
        
        # Save
        save_json(metrics, metrics_path)
        assert metrics_path.exists()
        
        # Load
        loaded_metrics = load_json(metrics_path)
        
        assert loaded_metrics["accuracy"] == 0.95
        assert loaded_metrics["roc_auc"] == 0.97
        assert loaded_metrics["report"]["REAL"]["f1-score"] == 0.95


class TestDataFlowIntegration:
    """Tests for data flow between components."""

    def test_text_clean_integration_with_training(self):
        """Test that text cleaning integrates correctly with training."""
        from text_clean import clean_text
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        
        # Raw texts with URLs, emails, etc.
        raw_texts = [
            "Visit https://gov.com for Policy Announcements!!! Contact info@gov.com",
            "SHOCKING!!! Conspiracy at http://fake.com - email lies@fake.com",
        ]
        labels = [0, 1]
        
        # Clean texts
        cleaned_texts = [clean_text(t) for t in raw_texts]
        
        # Train on cleaned texts
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20)),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ])
        pipe.fit(cleaned_texts, labels)
        
        # Predict on new cleaned text
        new_text = clean_text("Visit https://example.com for news")
        prob = pipe.predict_proba([new_text])[0, 1]
        
        assert 0 <= prob <= 1

    def test_csv_to_prediction_workflow(self, tmp_path):
        """Test complete workflow from CSV files to predictions."""
        from train_model import read_csv_any, ensure_dir
        from detect_fake_news import load_pipeline_or_parts
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        
        # Create CSV files
        real_data = pd.DataFrame({
            "title": ["Government News", "Science Discovery"],
            "text": [
                "The government passed new legislation today.",
                "Researchers found evidence of water on Mars."
            ],
            "subject": ["politics", "science"],
            "date": ["2024-01-15", "2024-01-14"],
        })
        
        fake_data = pd.DataFrame({
            "title": ["Conspiracy Alert", "Fake Scandal"],
            "text": [
                "Secret documents prove aliens control everything!",
                "Celebrity scandal that will shock you!"
            ],
            "subject": ["conspiracy", "entertainment"],
            "date": ["2024-01-15", "2024-01-14"],
        })
        
        real_path = tmp_path / "real.csv"
        fake_path = tmp_path / "fake.csv"
        real_data.to_csv(real_path, index=False)
        fake_data.to_csv(fake_path, index=False)
        
        # Load CSVs
        df_real = read_csv_any(real_path)
        df_fake = read_csv_any(fake_path)
        
        # Prepare combined text
        for df in (df_real, df_fake):
            title = df["title"].fillna("") if "title" in df.columns else ""
            txt = df.get("text", "").fillna("")
            df["combined_text"] = (title + " " + txt).str.strip()
        
        X = pd.concat([df_real["combined_text"], df_fake["combined_text"]], ignore_index=True)
        y = np.array([0] * len(df_real) + [1] * len(df_fake))
        
        # Train
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=30)),
            ("clf", RandomForestClassifier(n_estimators=10, random_state=42)),
        ])
        pipe.fit(X, y)
        
        # Save and reload
        outdir = ensure_dir(tmp_path / "output")
        pipeline_path = outdir / "pipeline.joblib"
        joblib.dump(pipe, pipeline_path)
        
        loaded_pipe, _, _ = load_pipeline_or_parts(str(pipeline_path), None, None)
        
        # Predict
        test_text = "Government announces new policy"
        prob = float(loaded_pipe.predict_proba([test_text])[0, 1])
        
        assert 0 <= prob <= 1


class TestComponentIntegration:
    """Tests for integration between specific components."""

    def test_utils_with_train_model(self, tmp_path):
        """Test utils module integration with training."""
        from utils import ensure_outdir, save_json, load_json
        from train_model import ensure_dir
        
        # Use utils to create output directory
        outdir = ensure_outdir(tmp_path / "utils_test")
        
        # Save metrics using utils
        metrics = {"accuracy": 0.92, "f1": 0.91}
        metrics_path = outdir / "metrics.json"
        save_json(metrics, metrics_path)
        
        # Load and verify
        loaded = load_json(metrics_path)
        assert loaded["accuracy"] == 0.92

    def test_text_clean_with_detect(self, sample_pipeline):
        """Test text cleaning integration with detection."""
        from text_clean import clean_text
        
        # Raw input with noise
        raw_input = "SHOCKING!!! Visit https://fake.com for CONSPIRACY details!!!"
        
        # Clean before prediction
        cleaned = clean_text(raw_input)
        
        # Predict on cleaned text
        prob = float(sample_pipeline.predict_proba([cleaned])[0, 1])
        assert 0 <= prob <= 1

    def test_pipeline_save_load_consistency(self, tmp_path):
        """Test that saved and loaded pipeline produces same predictions."""
        from train_model import ensure_dir
        from detect_fake_news import load_pipeline_or_parts
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        
        # Train
        train_texts = [
            "real news government policy",
            "science discovery research",
            "fake conspiracy shocking",
            "miracle cure secret",
        ]
        train_labels = [0, 0, 1, 1]
        
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20)),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ])
        pipe.fit(train_texts, train_labels)
        
        # Predict before saving
        test_text = "government policy announcement"
        prob_before = float(pipe.predict_proba([test_text])[0, 1])
        
        # Save and reload
        outdir = ensure_dir(tmp_path / "consistency_test")
        pipeline_path = outdir / "pipe.joblib"
        joblib.dump(pipe, pipeline_path)
        
        loaded_pipe, _, _ = load_pipeline_or_parts(str(pipeline_path), None, None)
        
        # Predict after loading
        prob_after = float(loaded_pipe.predict_proba([test_text])[0, 1])
        
        # Should be identical
        assert prob_before == prob_after


class TestErrorHandlingIntegration:
    """Tests for error handling in integrated workflows."""

    def test_missing_model_file_error(self, tmp_path):
        """Test error when model file doesn't exist."""
        from detect_fake_news import load_pipeline_or_parts
        
        with pytest.raises(Exception):  # FileNotFoundError or similar
            load_pipeline_or_parts(None, str(tmp_path / "nonexistent.joblib"), None)

    def test_corrupted_model_file_error(self, tmp_path):
        """Test error when model file is corrupted."""
        from detect_fake_news import load_pipeline_or_parts
        
        # Create a corrupted joblib file
        corrupted_path = tmp_path / "corrupted.joblib"
        with open(corrupted_path, "w") as f:
            f.write("not a valid joblib file")
        
        with pytest.raises(Exception):
            load_pipeline_or_parts(str(corrupted_path), None, None)

    def test_incompatible_vectorizer_model_pair(self, tmp_path):
        """Test error when vectorizer and model are incompatible."""
        from detect_fake_news import load_pipeline_or_parts
        
        # Create vectorizer fitted on different vocabulary
        vec1 = TfidfVectorizer()
        vec1.fit(["completely different vocabulary words"])
        
        # Create model trained on different features
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier()
        # Train on 10 features
        X_dummy = np.random.rand(10, 10)
        y_dummy = [0] * 5 + [1] * 5
        clf.fit(X_dummy, y_dummy)
        
        # Save both
        vec_path = tmp_path / "vec.joblib"
        model_path = tmp_path / "model.joblib"
        joblib.dump(vec1, vec_path)
        joblib.dump(clf, model_path)
        
        # Load both
        _, loaded_clf, loaded_vec = load_pipeline_or_parts(None, str(model_path), str(vec_path))
        
        # Transform with vectorizer
        X_test = loaded_vec.transform(["test text"])
        
        # This might produce unexpected results but shouldn't crash
        try:
            loaded_clf.predict(X_test)
        except Exception:
            # Expected due to feature mismatch
            pass
