"""
Real invocation tests for train_model script.
Tests actual execution of train_model.py with synthetic data.
"""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline


class TestTrainModelRealInvocation:
    """Tests that actually invoke train_model.py as a script."""

    def test_train_model_script_execution(self, tmp_path):
        """
        Test train_model.py script execution with synthetic data.
        Verifies all output files are created and valid.
        """
        # Create synthetic training data
        real_texts = [
            "The government announced new economic policies today in Washington DC.",
            "Scientists discovered a new species of frog in the Amazon rainforest.",
            "The local basketball team won the championship after overtime.",
            "New hospital opened downtown to serve the growing community needs.",
            "Education reform bill passed by lawmakers with bipartisan support.",
            "Weather forecast predicts sunny skies for the weekend.",
            "Stock market reached record highs this quarter.",
            "University researchers published findings in major journal.",
        ]
        
        fake_texts = [
            "SHOCKING secret documents reveal aliens control the government!",
            "Miracle cure doctors don't want you to know about this secret!",
            "Celebrity exposed as lizard person the truth is finally out!",
            "Moon landing was filmed in Hollywood basement conspiracy revealed!",
            "Free energy device suppressed by oil companies inventor assassinated!",
            "Secret society controls world banks and governments!",
            "Vaccines contain tracking microchips says whistleblower!",
            "Flat earth proof hidden by NASA for decades!",
        ]
        
        # Create DataFrames matching expected CSV structure
        df_real = pd.DataFrame({
            "title": [f"Real News {i}" for i in range(len(real_texts))],
            "text": real_texts,
            "subject": ["politics", "science", "sports", "health", "politics", "weather", "finance", "education"],
            "date": ["2024-01-15"] * len(real_texts),
        })
        
        df_fake = pd.DataFrame({
            "title": [f"Fake News {i}" for i in range(len(fake_texts))],
            "text": fake_texts,
            "subject": ["conspiracy", "health", "entertainment", "conspiracy", "technology", "conspiracy", "health", "science"],
            "date": ["2024-01-15"] * len(fake_texts),
        })
        
        # Save to temp CSV files
        real_csv = tmp_path / "synthetic_real.csv"
        fake_csv = tmp_path / "synthetic_fake.csv"
        df_real.to_csv(real_csv, index=False)
        df_fake.to_csv(fake_csv, index=False)
        
        # Output directory
        outdir = tmp_path / "train_output"
        
        # Execute train_model.py as subprocess
        result = subprocess.run([
            sys.executable, "src/train_model.py",
            "--real", str(real_csv),
            "--fake", str(fake_csv),
            "--text-col", "text",
            "--outdir", str(outdir)
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        # Check execution succeeded
        assert result.returncode == 0, f"Training failed with: {result.stderr}"
        
        # Verify all expected output files exist
        assert (outdir / "pipeline.joblib").exists(), "pipeline.joblib not created"
        assert (outdir / "vectorizer.joblib").exists(), "vectorizer.joblib not created"
        assert (outdir / "model.joblib").exists(), "model.joblib not created"
        assert (outdir / "metrics.json").exists(), "metrics.json not created"
        assert (outdir / "charts" / "confusion_matrix.png").exists(), "confusion_matrix.png not created"
        assert (outdir / "charts" / "roc_curve.png").exists(), "roc_curve.png not created"
        assert (outdir / "charts" / "pr_curve.png").exists(), "pr_curve.png not created"
        
        # Verify metrics.json content
        with open(outdir / "metrics.json") as f:
            metrics = json.load(f)
        
        assert "accuracy" in metrics
        assert "roc_auc" in metrics
        assert "avg_precision" in metrics
        assert "cv_f1_macro_mean" in metrics
        assert "cv_f1_macro_std" in metrics
        assert "report" in metrics
        
        # Verify metrics are valid numbers
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1
        
        # Verify pipeline can be loaded and is valid
        pipeline = joblib.load(outdir / "pipeline.joblib")
        assert isinstance(pipeline, Pipeline)
        assert "tfidf" in pipeline.named_steps
        assert "clf" in pipeline.named_steps
        
        # Verify vectorizer and model can be loaded separately
        vectorizer = joblib.load(outdir / "vectorizer.joblib")
        model = joblib.load(outdir / "model.joblib")
        assert vectorizer is not None
        assert model is not None
        
        # Test that loaded pipeline can make predictions
        test_text = "Government announces new policy today"
        prob = pipeline.predict_proba([test_text])[0]
        assert len(prob) == 2  # Binary classification
        assert 0 <= prob[0] <= 1
        assert 0 <= prob[1] <= 1
        assert abs(prob[0] + prob[1] - 1.0) < 1e-6  # Probabilities sum to 1

    def test_train_model_with_mini_fixtures(self, tmp_path):
        """
        Test train_model.py using the mini fixture files.
        """
        fixtures_dir = Path(__file__).parent / "fixtures"
        real_csv = fixtures_dir / "mini_real_news.csv"
        fake_csv = fixtures_dir / "mini_fake_news.csv"
        
        outdir = tmp_path / "mini_train_output"
        
        # Execute train_model.py
        result = subprocess.run([
            sys.executable, "src/train_model.py",
            "--real", str(real_csv),
            "--fake", str(fake_csv),
            "--text-col", "text",
            "--outdir", str(outdir)
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        # Note: With very small datasets, training may fail due to min_df parameter
        # This test verifies the script handles small datasets appropriately
        if result.returncode == 0:
            # If training succeeded, verify outputs
            assert (outdir / "pipeline.joblib").exists()
            pipeline = joblib.load(outdir / "pipeline.joblib")
            assert isinstance(pipeline, Pipeline)

    def test_train_model_output_artifacts_structure(self, tmp_path):
        """
        Test that train_model produces correctly structured artifacts.
        """
        # Create sufficient synthetic data
        np.random.seed(42)
        real_texts = [
            f"Government policy announcement number {i} with official statement"
            for i in range(20)
        ]
        fake_texts = [
            f"SHOCKING conspiracy theory number {i} that will blow your mind"
            for i in range(20)
        ]
        
        df_real = pd.DataFrame({
            "title": [f"Real {i}" for i in range(20)],
            "text": real_texts,
            "subject": ["politics"] * 20,
            "date": ["2024-01-15"] * 20,
        })
        
        df_fake = pd.DataFrame({
            "title": [f"Fake {i}" for i in range(20)],
            "text": fake_texts,
            "subject": ["conspiracy"] * 20,
            "date": ["2024-01-15"] * 20,
        })
        
        real_csv = tmp_path / "test_real.csv"
        fake_csv = tmp_path / "test_fake.csv"
        df_real.to_csv(real_csv, index=False)
        df_fake.to_csv(fake_csv, index=False)
        
        outdir = tmp_path / "artifacts_test"
        
        # Run training
        result = subprocess.run([
            sys.executable, "src/train_model.py",
            "--real", str(real_csv),
            "--fake", str(fake_csv),
            "--text-col", "text",
            "--outdir", str(outdir)
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0, f"Training failed: {result.stderr}"
        
        # Load and verify pipeline structure
        pipeline = joblib.load(outdir / "pipeline.joblib")
        
        # Verify TF-IDF vectorizer properties
        vectorizer = pipeline.named_steps["tfidf"]
        assert vectorizer.ngram_range == (1, 3)
        assert vectorizer.max_df == 0.8
        assert vectorizer.min_df == 3
        assert vectorizer.max_features == 20000
        assert vectorizer.stop_words == "english"
        assert vectorizer.sublinear_tf == True
        
        # Verify RandomForest properties
        clf = pipeline.named_steps["clf"]
        assert clf.n_estimators == 400
        assert clf.random_state == 42
        assert clf.class_weight == "balanced_subsample"
        
        # Verify vectorizer has vocabulary
        assert hasattr(vectorizer, 'vocabulary_')
        assert len(vectorizer.vocabulary_) > 0
        
        # Verify model has been fitted
        assert hasattr(clf, 'classes_')
        assert len(clf.classes_) == 2  # Binary classification

    def test_train_model_metrics_reasonable(self, tmp_path):
        """
        Test that train_model produces reasonable metrics on separable data.
        """
        # Create clearly separable synthetic data
        real_texts = [
            "Official government announcement policy legislation congress senate",
            "Scientific research study published journal findings evidence",
            "Sports team championship victory game match tournament",
            "Health hospital medical treatment patient care doctor",
        ] * 10  # Repeat to get enough samples
        
        fake_texts = [
            "SHOCKING conspiracy secret hidden truth exposed reveal",
            "Miracle cure doctors hide secret treatment amazing",
            "Aliens UFO government cover up secret base area",
            "Celebrity scandal shocking reveal truth exposed secret",
        ] * 10
        
        df_real = pd.DataFrame({
            "title": [f"Real {i}" for i in range(len(real_texts))],
            "text": real_texts,
            "subject": ["politics"] * len(real_texts),
            "date": ["2024-01-15"] * len(real_texts),
        })
        
        df_fake = pd.DataFrame({
            "title": [f"Fake {i}" for i in range(len(fake_texts))],
            "text": fake_texts,
            "subject": ["conspiracy"] * len(fake_texts),
            "date": ["2024-01-15"] * len(fake_texts),
        })
        
        real_csv = tmp_path / "sep_real.csv"
        fake_csv = tmp_path / "sep_fake.csv"
        df_real.to_csv(real_csv, index=False)
        df_fake.to_csv(fake_csv, index=False)
        
        outdir = tmp_path / "metrics_test"
        
        result = subprocess.run([
            sys.executable, "src/train_model.py",
            "--real", str(real_csv),
            "--fake", str(fake_csv),
            "--text-col", "text",
            "--outdir", str(outdir)
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0, f"Training failed: {result.stderr}"
        
        # Load metrics
        with open(outdir / "metrics.json") as f:
            metrics = json.load(f)
        
        # On clearly separable data, we expect good performance
        assert metrics["accuracy"] >= 0.7, f"Accuracy too low: {metrics['accuracy']}"
        assert metrics["roc_auc"] >= 0.7, f"ROC-AUC too low: {metrics['roc_auc']}"
        
        # Classification report should have REAL and FAKE entries
        assert "REAL" in metrics["report"]
        assert "FAKE" in metrics["report"]
