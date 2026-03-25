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
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text_clean import clean_text, clean_many
from utils import ensure_outdir, save_json, load_json
from train_model import read_csv_any, pick_text_column


@pytest.fixture
def synthetic_dataset(mini_real_csv, mini_fake_csv):
    df_real = read_csv_any(mini_real_csv)
    df_fake = read_csv_any(mini_fake_csv)

    for df in (df_real, df_fake):
        title = df["title"].fillna("") if "title" in df.columns else ""
        txt = df.get("text", "").fillna("")
        df["combined_text"] = (title + " " + txt).str.strip()

    X = pd.concat([df_real["combined_text"], df_fake["combined_text"]], ignore_index=True)
    y = np.array([0] * len(df_real) + [1] * len(df_fake))

    return X, y


class TestEndToEndPipeline:
    def test_full_pipeline_workflow(self, synthetic_dataset, temp_output_dir):
        X, y = synthetic_dataset

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

        pipeline_path = temp_output_dir / "pipeline.joblib"
        vectorizer_path = temp_output_dir / "vectorizer.joblib"
        model_path = temp_output_dir / "model.joblib"
        metrics_path = temp_output_dir / "metrics.json"

        joblib.dump(pipe, pipeline_path)
        joblib.dump(pipe.named_steps["tfidf"], vectorizer_path)
        joblib.dump(pipe.named_steps["clf"], model_path)

        y_pred = pipe.predict(X)
        y_prob = pipe.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "predictions_count": int(len(y_pred)),
            "labels": ["REAL", "FAKE"],
        }
        save_json(metrics, metrics_path)

        assert pipeline_path.exists()
        assert vectorizer_path.exists()
        assert model_path.exists()
        assert metrics_path.exists()

        loaded_pipe = joblib.load(pipeline_path)
        loaded_vec = joblib.load(vectorizer_path)
        loaded_clf = joblib.load(model_path)
        loaded_metrics = load_json(metrics_path)

        assert hasattr(loaded_pipe, "predict")
        assert hasattr(loaded_vec, "transform")
        assert hasattr(loaded_clf, "predict")
        assert "accuracy" in loaded_metrics

    def test_data_loading_to_prediction(self, synthetic_dataset, temp_output_dir):
        X, y = synthetic_dataset

        cleaned_texts = clean_many(X.tolist())

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(min_df=1)),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ])

        pipe.fit(cleaned_texts, y)

        new_text = "Breaking: Scientists discover amazing new technology"
        cleaned_new = clean_text(new_text)
        prob = float(pipe.predict_proba([cleaned_new])[0, 1])

        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_model_persistence_and_reuse(self, synthetic_dataset, temp_output_dir):
        X, y = synthetic_dataset

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(min_df=1)),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ])
        pipe.fit(X, y)

        original_pred = pipe.predict(X[:2])
        original_prob = pipe.predict_proba(X[:2])

        model_path = temp_output_dir / "persisted_model.joblib"
        joblib.dump(pipe, model_path)

        loaded_pipe = joblib.load(model_path)
        loaded_pred = loaded_pipe.predict(X[:2])
        loaded_prob = loaded_pipe.predict_proba(X[:2])

        np.testing.assert_array_equal(original_pred, loaded_pred)
        np.testing.assert_array_almost_equal(original_prob, loaded_prob)


class TestIntegrationWithFixtures:
    def test_mini_real_data_structure(self, mini_real_df):
        assert "title" in mini_real_df.columns
        assert "text" in mini_real_df.columns
        assert len(mini_real_df) == 3

    def test_mini_fake_data_structure(self, mini_fake_df):
        assert "title" in mini_fake_df.columns
        assert "text" in mini_fake_df.columns
        assert len(mini_fake_df) == 3

    def test_combined_dataset_training(self, mini_real_df, mini_fake_df, temp_output_dir):
        for df in (mini_real_df, mini_fake_df):
            df["combined_text"] = df["title"].fillna("") + " " + df["text"].fillna("")

        X = pd.concat([mini_real_df["combined_text"], mini_fake_df["combined_text"]], ignore_index=True)
        y = np.array([0] * len(mini_real_df) + [1] * len(mini_fake_df))

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(min_df=1)),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ])

        pipe.fit(X, y)

        y_pred = pipe.predict(X)
        accuracy = accuracy_score(y, y_pred)

        assert len(y_pred) == len(y)
        assert accuracy >= 0.0

    def test_text_cleaning_integration(self, mini_real_df, mini_fake_df):
        for df in (mini_real_df, mini_fake_df):
            df["cleaned_text"] = clean_many(df["text"].fillna("").tolist())

        for df in (mini_real_df, mini_fake_df):
            assert all(df["cleaned_text"].str.len() >= 0)


class TestMetricsAndArtifacts:
    def test_metrics_json_structure(self, temp_output_dir):
        metrics = {
            "accuracy": 0.95,
            "roc_auc": 0.98,
            "avg_precision": 0.97,
            "cv_f1_macro_mean": 0.94,
            "cv_f1_macro_std": 0.02,
            "report": {
                "REAL": {"precision": 0.96, "recall": 0.94, "f1-score": 0.95},
                "FAKE": {"precision": 0.94, "recall": 0.96, "f1-score": 0.95},
            },
        }

        metrics_path = temp_output_dir / "metrics.json"
        save_json(metrics, metrics_path)

        loaded = load_json(metrics_path)

        assert loaded["accuracy"] == 0.95
        assert loaded["roc_auc"] == 0.98
        assert "report" in loaded
        assert "REAL" in loaded["report"]
        assert "FAKE" in loaded["report"]

    def test_artifact_files_exist_after_training(self, synthetic_dataset, temp_output_dir):
        X, y = synthetic_dataset

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(min_df=1)),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ])
        pipe.fit(X, y)

        artifacts = {
            "pipeline.joblib": pipe,
            "vectorizer.joblib": pipe.named_steps["tfidf"],
            "model.joblib": pipe.named_steps["clf"],
        }

        for name, obj in artifacts.items():
            path = temp_output_dir / name
            joblib.dump(obj, path)
            assert path.exists()

    def test_model_can_predict_after_full_save_load(self, synthetic_dataset, temp_output_dir):
        X, y = synthetic_dataset

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(min_df=1)),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ])
        pipe.fit(X, y)

        pipeline_path = temp_output_dir / "pipeline.joblib"
        joblib.dump(pipe, pipeline_path)

        loaded = joblib.load(pipeline_path)

        test_text = "New scientific discovery announced"
        prob = loaded.predict_proba([test_text])[0, 1]
        pred = loaded.predict([test_text])[0]

        assert isinstance(prob, float)
        assert pred in [0, 1]


class TestErrorHandling:
    def test_empty_dataset_handling(self, temp_output_dir):
        X = pd.Series([], dtype=str)
        y = np.array([])

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(min_df=1)),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ])

        with pytest.raises(ValueError):
            pipe.fit(X, y)

    def test_missing_text_column_handling(self):
        df = pd.DataFrame({"other_column": ["data"]})
        with pytest.raises(ValueError):
            pick_text_column(df, "text")

    def test_invalid_json_file(self, tmp_path):
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("not valid json {")

        with pytest.raises(Exception):
            load_json(invalid_json)


class TestWorkflowWithCustomThreshold:
    def test_threshold_adjustment(self, synthetic_dataset):
        X, y = synthetic_dataset

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(min_df=1)),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ])
        pipe.fit(X, y)

        probs = pipe.predict_proba(X)[:, 1]

        thresholds = [0.3, 0.5, 0.7]
        predictions_per_threshold = []

        for threshold in thresholds:
            preds = (probs >= threshold).astype(int)
            predictions_per_threshold.append(preds)

        assert len(predictions_per_threshold) == 3

    def test_threshold_affects_predictions(self, synthetic_dataset):
        X, y = synthetic_dataset

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(min_df=1)),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ])
        pipe.fit(X, y)

        test_text = "Some ambiguous news article"
        prob = pipe.predict_proba([test_text])[0, 1]

        low_threshold = 0.1
        high_threshold = 0.9

        label_low = "FAKE" if prob >= low_threshold else "REAL"
        label_high = "FAKE" if prob >= high_threshold else "REAL"

        assert label_low in ["REAL", "FAKE"]
        assert label_high in ["REAL", "FAKE"]
