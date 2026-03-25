import sys
import json
import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_model import (
    ensure_dir,
    read_csv_any,
    pick_text_column,
    plot_confusion_matrix,
    plot_curve,
    LABELS,
)


class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "test_dir"
        result = ensure_dir(new_dir)
        assert new_dir.exists()
        assert result == new_dir

    def test_existing_directory(self, tmp_path):
        existing = tmp_path / "existing"
        existing.mkdir()
        result = ensure_dir(existing)
        assert existing.exists()

    def test_nested_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        result = ensure_dir(nested)
        assert nested.exists()


class TestReadCsvAny:
    def test_read_utf8_csv(self, mini_real_csv):
        df = read_csv_any(mini_real_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "title" in df.columns
        assert "text" in df.columns

    def test_read_with_nrows(self, mini_real_csv):
        df = read_csv_any(mini_real_csv, nrows=2)
        assert len(df) == 2

    def test_read_latin1_fallback(self, tmp_path):
        latin1_file = tmp_path / "latin1.csv"
        content = "title,text\nCafé,Résumé"
        latin1_file.write_bytes(content.encode("latin-1"))
        df = read_csv_any(latin1_file)
        assert len(df) == 1


class TestPickTextColumn:
    def test_preferred_column_exists(self):
        df = pd.DataFrame({"text": ["a"], "title": ["b"], "other": ["c"]})
        col = pick_text_column(df, "text")
        assert col == "text"

    def test_fallback_to_alternative(self):
        df = pd.DataFrame({"content": ["a"], "title": ["b"]})
        col = pick_text_column(df, "text")
        assert col == "content"

    def test_fallback_to_title(self):
        df = pd.DataFrame({"title": ["only title"]})
        col = pick_text_column(df, "text")
        assert col == "title"

    def test_raises_when_no_column(self):
        df = pd.DataFrame({"other": ["a"]})
        with pytest.raises(ValueError):
            pick_text_column(df, "text")


class TestPlotConfusionMatrix:
    def test_creates_file(self, tmp_path):
        cm = np.array([[10, 2], [3, 15]])
        out_file = tmp_path / "cm.png"
        plot_confusion_matrix(cm, out_file)
        assert out_file.exists()

    def test_with_custom_title(self, tmp_path):
        cm = np.array([[5, 0], [0, 5]])
        out_file = tmp_path / "cm_custom.png"
        plot_confusion_matrix(cm, out_file, title="Custom Title")
        assert out_file.exists()


class TestPlotCurve:
    def test_creates_roc_curve(self, tmp_path):
        x = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y = np.array([0.0, 0.5, 0.7, 0.85, 0.95, 1.0])
        out_file = tmp_path / "roc.png"
        plot_curve(x, y, out_file, "ROC", "FPR", "TPR")
        assert out_file.exists()

    def test_creates_pr_curve(self, tmp_path):
        x = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
        y = np.array([1.0, 0.9, 0.8, 0.7, 0.5])
        out_file = tmp_path / "pr.png"
        plot_curve(x, y, out_file, "PR Curve", "Recall", "Precision")
        assert out_file.exists()


class TestLabels:
    def test_labels_tuple(self):
        assert LABELS == ("REAL", "FAKE")


class TestTrainingOnMiniData:
    def test_mini_training_produces_pipeline(self, mini_real_csv, mini_fake_csv, temp_output_dir):
        df_real = read_csv_any(mini_real_csv)
        df_fake = read_csv_any(mini_fake_csv)

        for df in (df_real, df_fake):
            title = df["title"].fillna("") if "title" in df.columns else ""
            txt = df.get("text", "").fillna("")
            df["combined_text"] = (title + " " + txt).str.strip()

        X = pd.concat([df_real["combined_text"], df_fake["combined_text"]], ignore_index=True)
        y = np.array([0] * len(df_real) + [1] * len(df_fake))

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                sublinear_tf=True,
                stop_words="english",
                ngram_range=(1, 3),
                max_df=0.8,
                min_df=1,
                max_features=1000,
            )),
            ("clf", RandomForestClassifier(
                n_estimators=10,
                max_depth=5,
                random_state=42,
                n_jobs=1,
            )),
        ])

        pipe.fit(X, y)

        assert hasattr(pipe, "predict")
        assert hasattr(pipe, "predict_proba")

        pipeline_path = temp_output_dir / "pipeline.joblib"
        joblib.dump(pipe, pipeline_path)
        assert pipeline_path.exists()

        loaded_pipe = joblib.load(pipeline_path)
        assert hasattr(loaded_pipe, "predict")
        assert hasattr(loaded_pipe, "predict_proba")

    def test_mini_training_predictions(self, mini_real_csv, mini_fake_csv):
        df_real = read_csv_any(mini_real_csv)
        df_fake = read_csv_any(mini_fake_csv)

        for df in (df_real, df_fake):
            title = df["title"].fillna("") if "title" in df.columns else ""
            txt = df.get("text", "").fillna("")
            df["combined_text"] = (title + " " + txt).str.strip()

        X = pd.concat([df_real["combined_text"], df_fake["combined_text"]], ignore_index=True)
        y = np.array([0] * len(df_real) + [1] * len(df_fake))

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                sublinear_tf=True,
                stop_words="english",
                ngram_range=(1, 3),
                min_df=1,
                max_features=1000,
            )),
            ("clf", RandomForestClassifier(
                n_estimators=10,
                random_state=42,
                n_jobs=1,
            )),
        ])

        pipe.fit(X, y)
        predictions = pipe.predict(X)
        probabilities = pipe.predict_proba(X)

        assert len(predictions) == len(y)
        assert probabilities.shape[0] == len(y)
        assert probabilities.shape[1] == 2
        assert all(p in [0, 1] for p in predictions)

    def test_model_artifacts_structure(self, temp_output_dir, mini_real_csv, mini_fake_csv):
        df_real = read_csv_any(mini_real_csv)
        df_fake = read_csv_any(mini_fake_csv)

        for df in (df_real, df_fake):
            df["combined_text"] = df["title"].fillna("") + " " + df["text"].fillna("")

        X = pd.concat([df_real["combined_text"], df_fake["combined_text"]], ignore_index=True)
        y = np.array([0] * len(df_real) + [1] * len(df_fake))

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(min_df=1, max_features=100)),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ])
        pipe.fit(X, y)

        pipeline_path = temp_output_dir / "pipeline.joblib"
        vectorizer_path = temp_output_dir / "vectorizer.joblib"
        model_path = temp_output_dir / "model.joblib"
        metrics_path = temp_output_dir / "metrics.json"

        joblib.dump(pipe, pipeline_path)
        joblib.dump(pipe.named_steps["tfidf"], vectorizer_path)
        joblib.dump(pipe.named_steps["clf"], model_path)

        metrics = {
            "accuracy": 1.0,
            "roc_auc": 1.0,
            "report": {"REAL": {"precision": 1.0}, "FAKE": {"precision": 1.0}},
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

        assert pipeline_path.exists()
        assert vectorizer_path.exists()
        assert model_path.exists()
        assert metrics_path.exists()

        loaded_metrics = json.loads(metrics_path.read_text())
        assert "accuracy" in loaded_metrics
        assert "roc_auc" in loaded_metrics
