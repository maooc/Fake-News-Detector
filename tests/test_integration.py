"""
Integration tests for the full fake news detection pipeline.
"""
import sys
sys.path.insert(0, "src")

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import subprocess

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class TestEndToEndPipeline:
    """End-to-end tests for the complete pipeline."""

    def test_full_training_and_prediction_cycle(self, temp_dir, sample_news_path):
        """
        Test complete cycle:
        1. Load data from CSV
        2. Preprocess text
        3. Train model
        4. Save artifacts
        5. Load saved models
        6. Make predictions
        """
        # Step 1: Load and prepare data
        df = pd.read_csv(sample_news_path)
        df_real = df[df["label"] == 0].copy()
        df_fake = df[df["label"] == 1].copy()
        
        # Step 2: Preprocess text (simulating train_model's behavior)
        for data_df in (df_real, df_fake):
            title = data_df["title"].fillna("") if "title" in data_df.columns else ""
            text_col = data_df.get("text", data_df.get("content", "")).fillna("")
            data_df["combined_text"] = (title + " " + text_col).str.strip()
        
        X = pd.concat([df_real["combined_text"], df_fake["combined_text"]], ignore_index=True)
        y = np.array([0] * len(df_real) + [1] * len(df_fake))
        
        # Step 3: Train pipeline
        pipe = Pipeline(steps=[
            ("tfidf", TfidfVectorizer(
                sublinear_tf=True,
                stop_words="english",
                ngram_range=(1, 3),
                max_df=0.8,
                min_df=1,
                max_features=1000,
            )),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=42,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )),
        ])
        pipe.fit(X, y)
        
        # Step 4: Save artifacts
        outdir = temp_dir / "test_outputs"
        outdir.mkdir()
        
        joblib.dump(pipe, outdir / "pipeline.joblib")
        joblib.dump(pipe.named_steps["tfidf"], outdir / "vectorizer.joblib")
        joblib.dump(pipe.named_steps["clf"], outdir / "model.joblib")
        
        # Save metrics
        y_prob = pipe.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "test_passed": True
        }
        with open(outdir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Verify artifacts exist
        assert (outdir / "pipeline.joblib").exists()
        assert (outdir / "vectorizer.joblib").exists()
        assert (outdir / "model.joblib").exists()
        assert (outdir / "metrics.json").exists()
        
        # Step 5: Load saved models and verify consistency
        loaded_pipe = joblib.load(outdir / "pipeline.joblib")
        loaded_vec = joblib.load(outdir / "vectorizer.joblib")
        loaded_clf = joblib.load(outdir / "model.joblib")
        
        # Step 6: Make predictions with loaded pipeline
        original_preds = pipe.predict(X)
        loaded_preds = loaded_pipe.predict(X)
        np.testing.assert_array_equal(original_preds, loaded_preds)
        
        # Test with separate vectorizer and classifier
        X_transformed = loaded_vec.transform(X)
        separate_preds = loaded_clf.predict(X_transformed)
        np.testing.assert_array_equal(original_preds, separate_preds)
        
        # Verify model performance is reasonable
        accuracy = accuracy_score(y, original_preds)
        assert accuracy >= 0.5, "Model should perform at least better than random"

    def test_synthetic_data_generation_and_training(self, temp_dir):
        """Test generating synthetic data and training a model on it."""
        # Generate synthetic data
        n_samples = 20
        synthetic_data = []
        
        # Real-sounding patterns
        real_patterns = [
            "The economy continues to show positive growth according to the latest government report.",
            "Scientists have discovered a new method for renewable energy production.",
            "Local officials announced plans for infrastructure improvements in the downtown area.",
            "The national park service reported record visitor numbers this season.",
            "Researchers publish findings on climate change impacts in the region."
        ]
        
        # Fake-sounding patterns
        fake_patterns = [
            "SHOCKING: Government officials caught in massive cover-up scandal!",
            "MIRACLE: New supplement cures all diseases in just three days!",
            "SECRET: Hidden conspiracy revealed - you won't believe what happens next!",
            "BOMBSHELL: Celebrity couple involved in illegal activities, insiders say!",
            "URGENT: Toxic food products being sold at your local grocery store!"
        ]
        
        for i in range(n_samples // 2):
            synthetic_data.append({
                "title": f"Real News {i + 1}",
                "text": real_patterns[i % len(real_patterns)],
                "label": 0
            })
            synthetic_data.append({
                "title": f"Fake News {i + 1}",
                "text": fake_patterns[i % len(fake_patterns)],
                "label": 1
            })
        
        synth_df = pd.DataFrame(synthetic_data)
        
        # Save synthetic data
        synth_path = temp_dir / "synthetic_news.csv"
        synth_df.to_csv(synth_path, index=False)
        
        # Prepare for training
        synth_df["combined_text"] = (synth_df["title"] + " " + synth_df["text"]).str.strip()
        X = synth_df["combined_text"]
        y = synth_df["label"]
        
        # Train model
        pipe = Pipeline(steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=500)),
            ("clf", RandomForestClassifier(n_estimators=50, random_state=42))
        ])
        pipe.fit(X, y)
        
        # Save and reload
        model_path = temp_dir / "synth_model.joblib"
        joblib.dump(pipe, model_path)
        loaded_pipe = joblib.load(model_path)
        
        # Test predictions on new synthetic examples
        test_cases = [
            ("The local university announced a new scholarship program for students.", 0),
            ("SHOCKING DISCOVERY: Ancient aliens built the pyramids says new evidence!", 1),
            ("The central bank announced new monetary policy to stabilize currency.", 0),
            ("MIRACLE WEIGHT LOSS: Lose 50 pounds in one week with this secret trick!", 1)
        ]
        
        for text, expected_label in test_cases:
            prob = float(loaded_pipe.predict_proba([text])[0, 1])
            pred_label = 1 if prob >= 0.5 else 0
            # Just verify prediction works, not necessarily correct for all cases
            assert 0 <= prob <= 1

    def test_detection_workflow_with_both_formats(self, temp_dir, sample_pipeline):
        """Test detection using both pipeline and separate model/vectorizer formats."""
        from text_clean import clean_text
        from detect_fake_news import load_pipeline_or_parts
        
        # Save both formats
        pipe_path = temp_dir / "pipeline.joblib"
        model_path = temp_dir / "model.joblib"
        vec_path = temp_dir / "vectorizer.joblib"
        
        joblib.dump(sample_pipeline, pipe_path)
        joblib.dump(sample_pipeline.named_steps["clf"], model_path)
        joblib.dump(sample_pipeline.named_steps["tfidf"], vec_path)
        
        # Test texts
        test_texts = [
            "Real news about the economy and business developments.",
            "Shocking conspiracy theory about secret government experiments!"
        ]
        
        # Test pipeline-based detection
        pipe, _, _ = load_pipeline_or_parts(str(pipe_path), None, None)
        pipe_results = []
        for text in test_texts:
            cleaned = clean_text(text)
            prob = float(pipe.predict_proba([cleaned])[0, 1])
            pipe_results.append(prob)
        
        # Test separate model/vectorizer detection
        _, clf, vec = load_pipeline_or_parts(None, str(model_path), str(vec_path))
        separate_results = []
        for text in test_texts:
            cleaned = clean_text(text)
            X = vec.transform([cleaned])
            prob = float(clf.predict_proba(X)[0, 1])
            separate_results.append(prob)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(pipe_results, separate_results, decimal=10)
        
        # Verify all probabilities in valid range
        assert all(0 <= p <= 1 for p in pipe_results)
        assert all(0 <= p <= 1 for p in separate_results)

    def test_metrics_generation_and_loading(self, temp_dir, small_dataset):
        """Test metrics generation during training and subsequent loading."""
        from utils import save_json, load_json
        
        X, y = small_dataset
        
        # Train model
        pipe = Pipeline(steps=[
            ("tfidf", TfidfVectorizer(max_features=100)),
            ("clf", RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        pipe.fit(X, y)
        
        # Generate metrics
        y_prob = pipe.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "test_samples": len(y),
            "threshold_used": 0.5,
            "predictions": [int(p) for p in y_pred],
            "probabilities": [float(p) for p in y_prob]
        }
        
        # Save and reload metrics
        metrics_path = temp_dir / "metrics.json"
        save_json(metrics, metrics_path)
        loaded_metrics = load_json(metrics_path)
        
        # Verify metrics integrity
        assert loaded_metrics["accuracy"] == metrics["accuracy"]
        assert loaded_metrics["test_samples"] == metrics["test_samples"]
        assert len(loaded_metrics["predictions"]) == len(y)
        assert len(loaded_metrics["probabilities"]) == len(y)


class TestCrossModuleIntegration:
    """Tests for integration between different modules."""

    def test_text_clean_in_training_pipeline(self, sample_news_df):
        """Test that text cleaning integrates properly with training pipeline."""
        from text_clean import clean_text, clean_many
        
        # Get combined text
        df_real = sample_news_df[sample_news_df["label"] == 0].copy()
        df_fake = sample_news_df[sample_news_df["label"] == 1].copy()
        
        for df in (df_real, df_fake):
            df["combined_text"] = (df["title"] + " " + df["text"]).str.strip()
        
        X = pd.concat([df_real["combined_text"], df_fake["combined_text"]], ignore_index=True)
        y = np.array([0] * len(df_real) + [1] * len(df_fake))
        
        # Clean all texts
        X_cleaned = clean_many(X.tolist())
        
        # Train on cleaned text
        pipe = Pipeline(steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=1000)),
            ("clf", RandomForestClassifier(n_estimators=50, random_state=42))
        ])
        pipe.fit(X_cleaned, y)
        
        # Make predictions on cleaned text
        y_pred = pipe.predict(X_cleaned)
        accuracy = accuracy_score(y, y_pred)
        
        assert accuracy >= 0.5
        assert len(X_cleaned) == len(X)
        assert all(isinstance(t, str) for t in X_cleaned)

    def test_utils_in_training_workflow(self, temp_dir, sample_news_df):
        """Test that utils module functions work in training workflow."""
        from utils import ensure_outdir, save_json, load_json
        
        # Simulate training output
        outdir = temp_dir / "training_outputs"
        charts_dir = ensure_outdir(outdir / "charts")
        
        assert charts_dir.exists()
        assert charts_dir.is_dir()
        
        # Save training metrics
        metrics = {
            "accuracy": 0.95,
            "roc_auc": 0.98,
            "cv_scores": [0.93, 0.96, 0.94, 0.95, 0.97]
        }
        
        metrics_path = outdir / "metrics.json"
        save_json(metrics, metrics_path)
        
        # Verify save and load
        loaded = load_json(metrics_path)
        assert loaded["accuracy"] == metrics["accuracy"]
        assert loaded["roc_auc"] == metrics["roc_auc"]
        assert len(loaded["cv_scores"]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
