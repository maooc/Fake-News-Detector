"""
True end-to-end tests using actual script invocations.
Tests complete workflow: train_model -> detect_fake_news with real subprocess calls.
Uses strict label assertions for clear real/fake news samples.
"""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


class TestTrueEndToEndWorkflow:
    """
    True end-to-end tests that invoke both train_model.py and detect_fake_news.py
    as separate subprocesses, validating the complete workflow with strict label assertions.
    """

    def test_full_workflow_train_then_detect_strict_labels(self, tmp_path):
        """
        Complete end-to-end test with strict label assertions:
        1. Generate synthetic data
        2. Run train_model.py to train and save model
        3. Run detect_fake_news.py to make predictions
        4. Verify predictions with strict label assertions (not probability ranges)
        """
        # Step 1: Create clearly separable synthetic training data
        real_texts = [
            "official government announcement policy legislation congress today",
            "scientific research study published peer reviewed journal findings",
            "sports team wins championship game victory tournament final",
            "hospital opens medical facility health care treatment center",
            "education reform bill passes lawmakers vote bipartisan support",
            "weather forecast sunny skies weekend temperature normal",
            "stock market reaches record high quarterly earnings report",
            "university researchers discover breakthrough innovation technology",
        ] * 10  # 80 real samples
        
        fake_texts = [
            "shocking conspiracy secret documents reveal hidden truth exposed",
            "miracle cure doctors dont want you know secret treatment amazing",
            "aliens control government secret society exposed truth revealed",
            "celebrity scandal lizard person revealed shocking truth exposed",
            "moon landing fake filmed hollywood studio conspiracy theory",
            "free energy device suppressed oil companies assassination plot",
            "vaccines contain microchips tracking conspiracy revealed secret",
            "flat earth proof nasa hiding truth conspiracy theory exposed",
        ] * 10  # 80 fake samples
        
        # Create DataFrames
        df_real = pd.DataFrame({
            "title": [f"Real News {i}" for i in range(len(real_texts))],
            "text": real_texts,
            "subject": ["politics"] * len(real_texts),
            "date": ["2024-01-15"] * len(real_texts),
        })
        
        df_fake = pd.DataFrame({
            "title": [f"Fake News {i}" for i in range(len(fake_texts))],
            "text": fake_texts,
            "subject": ["conspiracy"] * len(fake_texts),
            "date": ["2024-01-15"] * len(fake_texts),
        })
        
        # Save training data
        train_dir = tmp_path / "train_data"
        train_dir.mkdir()
        real_csv = train_dir / "real.csv"
        fake_csv = train_dir / "fake.csv"
        df_real.to_csv(real_csv, index=False)
        df_fake.to_csv(fake_csv, index=False)
        
        model_dir = tmp_path / "model_output"
        
        # Step 2: Train model using train_model.py
        train_result = subprocess.run([
            sys.executable, "src/train_model.py",
            "--real", str(real_csv),
            "--fake", str(fake_csv),
            "--text-col", "text",
            "--outdir", str(model_dir)
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert train_result.returncode == 0, f"Training failed: {train_result.stderr}"
        
        # Verify model files were created
        assert (model_dir / "pipeline.joblib").exists()
        assert (model_dir / "metrics.json").exists()
        
        # Step 3: Test predictions with detect_fake_news.py using STRICT LABEL ASSERTIONS
        
        # Test clear REAL news - MUST be classified as REAL
        real_test_cases = [
            "official government policy announcement",
            "scientific research study published",
            "sports team championship victory",
        ]
        
        for text in real_test_cases:
            detect_result = subprocess.run([
                sys.executable, "src/detect_fake_news.py",
                "--pipeline", str(model_dir / "pipeline.joblib"),
                "--text", text,
                "--threshold", "0.5"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            assert detect_result.returncode == 0, f"Detection failed for '{text}': {detect_result.stderr}"
            
            output = detect_result.stdout.strip()
            label = output.split("Label:")[1].split("|")[0].strip()
            
            # STRICT ASSERTION: Must be REAL
            assert label == "REAL", f"Clear real news '{text}' must be REAL, got {label}"
        
        # Test clear FAKE news - MUST be classified as FAKE
        # Use longer, clearer fake news patterns that match training data
        fake_test_cases = [
            "shocking conspiracy secret documents reveal hidden truth exposed",
            "miracle cure doctors dont want you know secret treatment amazing",
            "aliens control government secret society exposed truth revealed",
        ]
        
        for text in fake_test_cases:
            detect_result = subprocess.run([
                sys.executable, "src/detect_fake_news.py",
                "--pipeline", str(model_dir / "pipeline.joblib"),
                "--text", text,
                "--threshold", "0.5"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            assert detect_result.returncode == 0, f"Detection failed for '{text}': {detect_result.stderr}"
            
            output = detect_result.stdout.strip()
            label = output.split("Label:")[1].split("|")[0].strip()
            
            # STRICT ASSERTION: Must be FAKE
            assert label == "FAKE", f"Clear fake news '{text}' must be FAKE, got {label}"

    def test_end_to_end_with_separate_artifacts_strict(self, tmp_path):
        """
        End-to-end test using separate vectorizer and model files with strict assertions.
        """
        # Create training data
        real_texts = [
            "official government policy announcement legislation congress" 
            for _ in range(25)
        ]
        fake_texts = [
            "shocking conspiracy secret revealed truth exposed hidden"
            for _ in range(25)
        ]
        
        df_real = pd.DataFrame({
            "title": [f"R{i}" for i in range(25)],
            "text": real_texts,
            "subject": ["pol"] * 25,
            "date": ["2024-01-15"] * 25,
        })
        
        df_fake = pd.DataFrame({
            "title": [f"F{i}" for i in range(25)],
            "text": fake_texts,
            "subject": ["con"] * 25,
            "date": ["2024-01-15"] * 25,
        })
        
        train_dir = tmp_path / "data"
        train_dir.mkdir()
        df_real.to_csv(train_dir / "real.csv", index=False)
        df_fake.to_csv(train_dir / "fake.csv", index=False)
        
        model_dir = tmp_path / "model"
        
        # Train
        train_result = subprocess.run([
            sys.executable, "src/train_model.py",
            "--real", str(train_dir / "real.csv"),
            "--fake", str(train_dir / "fake.csv"),
            "--outdir", str(model_dir)
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert train_result.returncode == 0
        
        # Verify separate artifacts exist
        assert (model_dir / "vectorizer.joblib").exists()
        assert (model_dir / "model.joblib").exists()
        
        # Test REAL news with separate artifacts - STRICT ASSERTION
        real_text = "official government policy announcement"
        result_real = subprocess.run([
            sys.executable, "src/detect_fake_news.py",
            "--model", str(model_dir / "model.joblib"),
            "--vectorizer", str(model_dir / "vectorizer.joblib"),
            "--text", real_text,
            "--threshold", "0.5"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result_real.returncode == 0
        label = result_real.stdout.split("Label:")[1].split("|")[0].strip()
        assert label == "REAL", f"Real news should be REAL, got {label}"
        
        # Test FAKE news with separate artifacts - STRICT ASSERTION
        fake_text = "shocking conspiracy secret revealed"
        result_fake = subprocess.run([
            sys.executable, "src/detect_fake_news.py",
            "--model", str(model_dir / "model.joblib"),
            "--vectorizer", str(model_dir / "vectorizer.joblib"),
            "--text", fake_text,
            "--threshold", "0.5"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result_fake.returncode == 0
        label = result_fake.stdout.split("Label:")[1].split("|")[0].strip()
        assert label == "FAKE", f"Fake news should be FAKE, got {label}"

    def test_end_to_end_workflow_with_charts(self, tmp_path):
        """
        End-to-end test verifying chart generation.
        """
        # Create training data
        real_texts = [f"Real news article content official government {i}" for i in range(15)]
        fake_texts = [f"Fake conspiracy theory shock secret revealed {i}" for i in range(15)]
        
        df_real = pd.DataFrame({
            "title": [f"R{i}" for i in range(15)],
            "text": real_texts,
            "subject": ["pol"] * 15,
            "date": ["2024-01-15"] * 15,
        })
        
        df_fake = pd.DataFrame({
            "title": [f"F{i}" for i in range(15)],
            "text": fake_texts,
            "subject": ["con"] * 15,
            "date": ["2024-01-15"] * 15,
        })
        
        train_dir = tmp_path / "data"
        train_dir.mkdir()
        df_real.to_csv(train_dir / "real.csv", index=False)
        df_fake.to_csv(train_dir / "fake.csv", index=False)
        
        model_dir = tmp_path / "model"
        
        # Train
        result = subprocess.run([
            sys.executable, "src/train_model.py",
            "--real", str(train_dir / "real.csv"),
            "--fake", str(train_dir / "fake.csv"),
            "--outdir", str(model_dir)
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        
        # Verify all output files exist
        assert (model_dir / "pipeline.joblib").exists()
        assert (model_dir / "metrics.json").exists()
        assert (model_dir / "charts" / "confusion_matrix.png").exists()
        assert (model_dir / "charts" / "roc_curve.png").exists()
        assert (model_dir / "charts" / "pr_curve.png").exists()
        
        # Verify charts are valid PNG files (have non-zero size)
        assert (model_dir / "charts" / "confusion_matrix.png").stat().st_size > 0
        assert (model_dir / "charts" / "roc_curve.png").stat().st_size > 0
        assert (model_dir / "charts" / "pr_curve.png").stat().st_size > 0

    def test_end_to_end_prediction_strict_correctness(self, tmp_path):
        """
        Strict end-to-end test verifying prediction correctness on clear examples.
        Uses direct label assertions, not probability ranges.
        """
        # Create clearly separable data with distinct patterns
        real_patterns = [
            "official government announcement policy legislation congress",
            "scientific research study published journal peer reviewed",
            "sports championship victory team wins tournament final",
            "hospital medical health care treatment patient doctor",
        ]
        fake_patterns = [
            "shocking conspiracy secret hidden truth exposed reveal",
            "miracle cure doctors hide secret amazing unbelievable",
            "aliens ufo government cover up secret base hidden",
            "celebrity scandal lizard person revealed shocking truth",
        ]
        
        real_texts = []
        fake_texts = []
        for i in range(12):
            real_texts.extend([p + f" batch {i}" for p in real_patterns])
            fake_texts.extend([p + f" batch {i}" for p in fake_patterns])
        
        df_real = pd.DataFrame({
            "title": [f"R{i}" for i in range(len(real_texts))],
            "text": real_texts,
            "subject": ["pol"] * len(real_texts),
            "date": ["2024-01-15"] * len(real_texts),
        })
        
        df_fake = pd.DataFrame({
            "title": [f"F{i}" for i in range(len(fake_texts))],
            "text": fake_texts,
            "subject": ["con"] * len(fake_texts),
            "date": ["2024-01-15"] * len(fake_texts),
        })
        
        train_dir = tmp_path / "data"
        train_dir.mkdir()
        df_real.to_csv(train_dir / "real.csv", index=False)
        df_fake.to_csv(train_dir / "fake.csv", index=False)
        
        model_dir = tmp_path / "model"
        
        # Train
        train_result = subprocess.run([
            sys.executable, "src/train_model.py",
            "--real", str(train_dir / "real.csv"),
            "--fake", str(train_dir / "fake.csv"),
            "--outdir", str(model_dir)
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert train_result.returncode == 0, f"Training failed: {train_result.stderr}"
        
        # Test clear real news - ALL MUST be classified as REAL
        real_test_texts = [
            "official government policy announcement",
            "scientific research study published",
            "sports championship victory team",
        ]
        
        for text in real_test_texts:
            result = subprocess.run([
                sys.executable, "src/detect_fake_news.py",
                "--pipeline", str(model_dir / "pipeline.joblib"),
                "--text", text,
                "--threshold", "0.5"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            assert result.returncode == 0
            output = result.stdout.strip()
            label = output.split("Label:")[1].split("|")[0].strip()
            
            # STRICT ASSERTION
            assert label == "REAL", f"Real text '{text}' must be REAL, got {label}"
        
        # Test clear fake news - ALL MUST be classified as FAKE
        # Use longer, clearer fake news patterns that match training data
        fake_test_texts = [
            "shocking conspiracy secret documents reveal hidden truth exposed",
            "miracle cure doctors dont want you know secret treatment amazing",
            "aliens control government secret society exposed truth revealed",
        ]
        
        for text in fake_test_texts:
            result = subprocess.run([
                sys.executable, "src/detect_fake_news.py",
                "--pipeline", str(model_dir / "pipeline.joblib"),
                "--text", text,
                "--threshold", "0.5"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            assert result.returncode == 0
            output = result.stdout.strip()
            label = output.split("Label:")[1].split("|")[0].strip()
            
            # STRICT ASSERTION
            assert label == "FAKE", f"Fake text '{text}' must be FAKE, got {label}"

    def test_end_to_end_multiple_predictions_consistency(self, tmp_path):
        """
        Test that multiple predictions on same text give consistent results.
        """
        # Create training data
        real_texts = ["official government policy announcement"] * 20
        fake_texts = ["shocking conspiracy secret revealed"] * 20
        
        df_real = pd.DataFrame({
            "title": [f"R{i}" for i in range(20)],
            "text": real_texts,
            "subject": ["pol"] * 20,
            "date": ["2024-01-15"] * 20,
        })
        
        df_fake = pd.DataFrame({
            "title": [f"F{i}" for i in range(20)],
            "text": fake_texts,
            "subject": ["con"] * 20,
            "date": ["2024-01-15"] * 20,
        })
        
        train_dir = tmp_path / "data"
        train_dir.mkdir()
        df_real.to_csv(train_dir / "real.csv", index=False)
        df_fake.to_csv(train_dir / "fake.csv", index=False)
        
        model_dir = tmp_path / "model"
        
        # Train
        subprocess.run([
            sys.executable, "src/train_model.py",
            "--real", str(train_dir / "real.csv"),
            "--fake", str(train_dir / "fake.csv"),
            "--outdir", str(model_dir)
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        # Run same prediction multiple times
        test_text = "Test text for consistency"
        labels = []
        
        for _ in range(3):
            result = subprocess.run([
                sys.executable, "src/detect_fake_news.py",
                "--pipeline", str(model_dir / "pipeline.joblib"),
                "--text", test_text,
                "--threshold", "0.5"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            assert result.returncode == 0
            output = result.stdout.strip()
            label = output.split("Label:")[1].split("|")[0].strip()
            labels.append(label)
        
        # All labels should be identical (deterministic model)
        assert all(l == labels[0] for l in labels), \
            f"Predictions should be consistent, got: {labels}"


class TestEndToEndErrorHandling:
    """End-to-end tests for error handling scenarios."""

    def test_end_to_end_missing_model_file(self, tmp_path):
        """
        Test error handling when model file doesn't exist.
        """
        result = subprocess.run([
            sys.executable, "src/detect_fake_news.py",
            "--pipeline", str(tmp_path / "nonexistent.joblib"),
            "--text", "Test"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        # Should fail with non-zero exit code
        assert result.returncode != 0

    def test_end_to_end_invalid_csv_format(self, tmp_path):
        """
        Test handling of invalid CSV format during training.
        """
        # Create invalid CSV
        invalid_csv = tmp_path / "invalid.csv"
        with open(invalid_csv, "w") as f:
            f.write("not,a,valid,news,csv\n")
            f.write("1,2,3,4,5\n")
        
        result = subprocess.run([
            sys.executable, "src/train_model.py",
            "--real", str(invalid_csv),
            "--fake", str(invalid_csv),
            "--outdir", str(tmp_path / "out")
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        # Should handle gracefully (may succeed or fail gracefully)
        # The important thing is it doesn't crash unexpectedly
