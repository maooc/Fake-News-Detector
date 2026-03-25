"""
Real invocation tests for detect_fake_news script.
Tests actual execution of detect_fake_news.py with strict assertions on prediction correctness.
Uses models trained by train_model.py for strict classification validation.
"""

import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline


class TestDetectFakeNewsWithTrainModelArtifacts:
    """
    Tests detect_fake_news.py using models actually trained by train_model.py.
    These tests use subprocess calls to train_model to generate real artifacts,
    then test detect_fake_news with strict label assertions.
    """

    @pytest.fixture
    def train_model_artifacts(self, tmp_path):
        """
        Generate model artifacts by actually running train_model.py.
        Returns path to output directory containing pipeline.joblib, etc.
        """
        # Create clearly separable training data
        real_texts = [
            "official government announcement policy legislation congress senate",
            "scientific research study published journal peer reviewed findings",
            "sports team wins championship game victory tournament final match",
            "hospital opens medical facility health care treatment patient",
            "education reform bill passes lawmakers vote bipartisan support",
            "weather forecast sunny skies temperature normal seasonal",
            "stock market reaches record high quarterly earnings growth",
            "university researchers discover breakthrough innovation technology",
        ] * 8  # 64 real samples
        
        fake_texts = [
            "shocking conspiracy secret documents reveal hidden truth exposed",
            "miracle cure doctors dont want you know secret treatment amazing",
            "aliens control government secret society exposed truth revealed",
            "celebrity scandal lizard person revealed shocking truth exposed",
            "moon landing fake filmed hollywood studio conspiracy theory",
            "free energy device suppressed oil companies assassination plot",
            "vaccines contain microchips tracking conspiracy revealed secret",
            "flat earth proof nasa hiding truth conspiracy theory exposed",
        ] * 8  # 64 fake samples
        
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
        
        # Output directory for model artifacts
        model_dir = tmp_path / "model_output"
        
        # Run train_model.py
        result = subprocess.run([
            sys.executable, "src/train_model.py",
            "--real", str(real_csv),
            "--fake", str(fake_csv),
            "--text-col", "text",
            "--outdir", str(model_dir)
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0, f"Training failed: {result.stderr}"
        assert (model_dir / "pipeline.joblib").exists()
        assert (model_dir / "vectorizer.joblib").exists()
        assert (model_dir / "model.joblib").exists()
        
        return model_dir

    def test_real_news_strict_label_assertion(self, train_model_artifacts):
        """
        Strict test: Clear real news MUST be classified as REAL.
        Uses model trained by train_model.py.
        """
        # Clear real news samples
        real_texts = [
            "official government announcement policy legislation",
            "scientific research study published journal",
            "sports team wins championship victory",
        ]
        
        for text in real_texts:
            result = subprocess.run([
                sys.executable, "src/detect_fake_news.py",
                "--pipeline", str(train_model_artifacts / "pipeline.joblib"),
                "--text", text,
                "--threshold", "0.5"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            assert result.returncode == 0, f"Detection failed for: {text}"
            
            output = result.stdout.strip()
            label = output.split("Label:")[1].split("|")[0].strip()
            
            # STRICT ASSERTION: Must be REAL
            assert label == "REAL", f"Expected REAL for '{text}', got {label}"

    def test_fake_news_strict_label_assertion(self, train_model_artifacts):
        """
        Strict test: Clear fake news MUST be classified as FAKE.
        Uses model trained by train_model.py.
        """
        # Clear fake news samples
        fake_texts = [
            "shocking conspiracy secret documents reveal",
            "miracle cure doctors dont want you know",
            "aliens control government secret society",
        ]
        
        for text in fake_texts:
            result = subprocess.run([
                sys.executable, "src/detect_fake_news.py",
                "--pipeline", str(train_model_artifacts / "pipeline.joblib"),
                "--text", text,
                "--threshold", "0.5"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            assert result.returncode == 0, f"Detection failed for: {text}"
            
            output = result.stdout.strip()
            label = output.split("Label:")[1].split("|")[0].strip()
            
            # STRICT ASSERTION: Must be FAKE
            assert label == "FAKE", f"Expected FAKE for '{text}', got {label}"

    def test_real_news_with_separate_artifacts(self, train_model_artifacts):
        """
        Test using separate vectorizer and model files trained by train_model.py.
        Strict assertion on real news classification.
        """
        text = "government official announcement policy legislation today"
        
        result = subprocess.run([
            sys.executable, "src/detect_fake_news.py",
            "--model", str(train_model_artifacts / "model.joblib"),
            "--vectorizer", str(train_model_artifacts / "vectorizer.joblib"),
            "--text", text,
            "--threshold", "0.5"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        
        output = result.stdout.strip()
        label = output.split("Label:")[1].split("|")[0].strip()
        
        # STRICT ASSERTION
        assert label == "REAL", f"Expected REAL, got {label}"

    def test_fake_news_with_separate_artifacts(self, train_model_artifacts):
        """
        Test using separate vectorizer and model files trained by train_model.py.
        Strict assertion on fake news classification.
        """
        text = "shocking conspiracy secret revealed truth exposed"
        
        result = subprocess.run([
            sys.executable, "src/detect_fake_news.py",
            "--model", str(train_model_artifacts / "model.joblib"),
            "--vectorizer", str(train_model_artifacts / "vectorizer.joblib"),
            "--text", text,
            "--threshold", "0.5"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        
        output = result.stdout.strip()
        label = output.split("Label:")[1].split("|")[0].strip()
        
        # STRICT ASSERTION
        assert label == "FAKE", f"Expected FAKE, got {label}"

    def test_multiple_real_samples_all_classified_real(self, train_model_artifacts):
        """
        Test multiple real news samples - ALL must be classified as REAL.
        """
        real_texts = [
            "official government announcement",
            "scientific research published",
            "sports championship victory",
            "hospital medical treatment",
            "education reform bill passes",
        ]
        
        for text in real_texts:
            result = subprocess.run([
                sys.executable, "src/detect_fake_news.py",
                "--pipeline", str(train_model_artifacts / "pipeline.joblib"),
                "--text", text,
                "--threshold", "0.5"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            assert result.returncode == 0
            
            output = result.stdout.strip()
            label = output.split("Label:")[1].split("|")[0].strip()
            
            # STRICT ASSERTION: Every single one must be REAL
            assert label == "REAL", f"Sample '{text}' should be REAL, got {label}"

    def test_multiple_fake_samples_all_classified_fake(self, train_model_artifacts):
        """
        Test multiple fake news samples - ALL must be classified as FAKE.
        Uses longer, clearer fake news patterns that match training data.
        """
        fake_texts = [
            "shocking conspiracy secret documents reveal hidden truth",
            "miracle cure doctors dont want you know secret treatment",
            "aliens control government secret society exposed truth",
            "celebrity scandal lizard person revealed shocking truth",
            "moon landing fake filmed hollywood studio conspiracy theory",
        ]
        
        for text in fake_texts:
            result = subprocess.run([
                sys.executable, "src/detect_fake_news.py",
                "--pipeline", str(train_model_artifacts / "pipeline.joblib"),
                "--text", text,
                "--threshold", "0.5"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            assert result.returncode == 0
            
            output = result.stdout.strip()
            label = output.split("Label:")[1].split("|")[0].strip()
            
            # STRICT ASSERTION: Every single one must be FAKE
            assert label == "FAKE", f"Sample '{text}' should be FAKE, got {label}"


class TestDetectFakeNewsOutputValidation:
    """Tests for output format and basic functionality."""

    @pytest.fixture
    def trained_model_dir(self, tmp_path):
        """Create a trained model using train_model.py."""
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
        
        result = subprocess.run([
            sys.executable, "src/train_model.py",
            "--real", str(train_dir / "real.csv"),
            "--fake", str(train_dir / "fake.csv"),
            "--outdir", str(model_dir)
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        return model_dir

    def test_detect_output_format(self, trained_model_dir):
        """Test that detect_fake_news.py output format is correct."""
        text = "Test news article content"
        
        result = subprocess.run([
            sys.executable, "src/detect_fake_news.py",
            "--pipeline", str(trained_model_dir / "pipeline.joblib"),
            "--text", text,
            "--threshold", "0.5"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        
        output = result.stdout.strip()
        
        # Verify format
        assert output.startswith("Label:"), f"Output should start with 'Label:', got: {output}"
        assert "Fake probability:" in output
        assert "Threshold:" in output
        
        # Verify label is either REAL or FAKE
        label = output.split("Label:")[1].split("|")[0].strip()
        assert label in ["REAL", "FAKE"], f"Label should be REAL or FAKE, got: {label}"
        
        # Verify probability is valid
        prob_str = output.split("Fake probability:")[1].split("|")[0].strip()
        prob = float(prob_str)
        assert 0 <= prob <= 1

    def test_detect_with_empty_text(self, trained_model_dir):
        """Test detect_fake_news.py with empty text."""
        result = subprocess.run([
            sys.executable, "src/detect_fake_news.py",
            "--pipeline", str(trained_model_dir / "pipeline.joblib"),
            "--text", "",
            "--threshold", "0.5"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0, f"Should handle empty text, got: {result.stderr}"
        
        output = result.stdout.strip()
        assert "Label:" in output
        assert "Fake probability:" in output

    def test_detect_with_custom_threshold(self, trained_model_dir):
        """Test detect_fake_news.py with custom threshold."""
        text = "government official announcement policy legislation today"
        
        # Test with high threshold (0.8) - should favor REAL
        result = subprocess.run([
            sys.executable, "src/detect_fake_news.py",
            "--pipeline", str(trained_model_dir / "pipeline.joblib"),
            "--text", text,
            "--threshold", "0.8"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        output = result.stdout.strip()
        label = output.split("Label:")[1].split("|")[0].strip()
        
        # With high threshold, this should be REAL
        assert label == "REAL", f"With threshold 0.8, expected REAL, got {label}"
