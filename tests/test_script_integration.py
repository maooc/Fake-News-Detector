import sys
import json
import subprocess
from pathlib import Path
import pytest
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestTrainModelScriptReal:
    def test_train_model_script_generates_all_artifacts(self, tmp_path):
        real_csv = tmp_path / "real.csv"
        fake_csv = tmp_path / "fake.csv"
        outdir = tmp_path / "outputs"

        real_data = pd.DataFrame({
            "title": [
                "Scientists Discover New Species",
                "Government Reports Economic Growth",
                "Local Election Results Announced",
                "Research Shows Climate Change Impact",
                "New Medical Treatment Approved",
            ],
            "text": [
                "Scientists have discovered a new species in the Amazon rainforest.",
                "The economy grew by 3% in the last quarter according to official reports.",
                "City officials released final election results yesterday.",
                "New research indicates significant climate change effects on agriculture.",
                "FDA approved a new treatment for rare diseases after clinical trials.",
            ],
            "subject": ["science", "economics", "politics", "science", "health"],
            "date": ["2024-01-01"] * 5,
        })
        real_data.to_csv(real_csv, index=False)

        fake_data = pd.DataFrame({
            "title": [
                "SHOCKING: Government Hiding Aliens!",
                "Doctors HATE This One Trick!",
                "You Won't BELIEVE What Happened!",
                "MIRACLE Cure They Don't Want You To Know!",
                "CELEBRITY Secret REVEALED!",
            ],
            "text": [
                "The government has been hiding alien contact for decades! Sources reveal the truth!",
                "Doctors don't want you to know this simple trick that cures everything!",
                "Shocking details that will blow your mind! Click here for more!",
                "Big Pharma is hiding this miracle cure from the public! Read more!",
                "Hollywood doesn't want you to know this celebrity secret! Shocking!",
            ],
            "subject": ["conspiracy", "health", "entertainment", "health", "entertainment"],
            "date": ["2024-01-01"] * 5,
        })
        fake_data.to_csv(fake_csv, index=False)

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "train_model.py"),
                "--real", str(real_csv),
                "--fake", str(fake_csv),
                "--text-col", "text",
                "--outdir", str(outdir),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0, f"Training failed: {result.stderr}"

        assert (outdir / "pipeline.joblib").exists()
        assert (outdir / "model.joblib").exists()
        assert (outdir / "vectorizer.joblib").exists()
        assert (outdir / "metrics.json").exists()

        pipeline = joblib.load(outdir / "pipeline.joblib")
        assert hasattr(pipeline, "predict")
        assert hasattr(pipeline, "predict_proba")

        model = joblib.load(outdir / "model.joblib")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

        vectorizer = joblib.load(outdir / "vectorizer.joblib")
        assert hasattr(vectorizer, "transform")

        with open(outdir / "metrics.json") as f:
            metrics = json.load(f)
        assert "accuracy" in metrics
        assert "roc_auc" in metrics
        assert "report" in metrics
        assert metrics["accuracy"] >= 0.0

    def test_train_model_script_generates_charts(self, tmp_path):
        real_csv = tmp_path / "real.csv"
        fake_csv = tmp_path / "fake.csv"
        outdir = tmp_path / "outputs"

        real_data = pd.DataFrame({
            "title": ["News Title " + str(i) for i in range(10)],
            "text": ["Real news content number " + str(i) for i in range(10)],
            "subject": ["news"] * 10,
            "date": ["2024-01-01"] * 10,
        })
        real_data.to_csv(real_csv, index=False)

        fake_data = pd.DataFrame({
            "title": ["SHOCKING Title " + str(i) for i in range(10)],
            "text": ["Fake news clickbait content " + str(i) for i in range(10)],
            "subject": ["fake"] * 10,
            "date": ["2024-01-01"] * 10,
        })
        fake_data.to_csv(fake_csv, index=False)

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "train_model.py"),
                "--real", str(real_csv),
                "--fake", str(fake_csv),
                "--outdir", str(outdir),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0

        charts_dir = outdir / "charts"
        assert (charts_dir / "confusion_matrix.png").exists()
        assert (charts_dir / "roc_curve.png").exists()
        assert (charts_dir / "pr_curve.png").exists()

    def test_train_model_script_handles_small_dataset(self, tmp_path):
        real_csv = tmp_path / "real.csv"
        fake_csv = tmp_path / "fake.csv"
        outdir = tmp_path / "outputs"

        real_data = pd.DataFrame({
            "title": [
                "Real News One",
                "Real News Two",
                "Real News Three",
                "Real News Four",
                "Real News Five",
            ],
            "text": [
                "This is a legitimate news article about science and research.",
                "Government officials announced new policies today in a press conference.",
                "The research team published their findings in a scientific journal.",
                "Local authorities confirmed the details of the new initiative.",
                "Experts have validated the data through rigorous analysis.",
            ],
            "subject": ["science"] * 5,
            "date": ["2024-01-01"] * 5,
        })
        real_data.to_csv(real_csv, index=False)

        fake_data = pd.DataFrame({
            "title": [
                "SHOCKING News One!",
                "SHOCKING News Two!",
                "SHOCKING News Three!",
                "SHOCKING News Four!",
                "SHOCKING News Five!",
            ],
            "text": [
                "SHOCKING revelation you won't believe! Click here now!",
                "Doctors HATE this one simple trick! Must read immediately!",
                "You won't BELIEVE what happened! Mind-blowing content!",
                "MIRACLE cure they're hiding from you! Share this now!",
                "The TRUTH they don't want you to know! Shocking details!",
            ],
            "subject": ["fake"] * 5,
            "date": ["2024-01-01"] * 5,
        })
        fake_data.to_csv(fake_csv, index=False)

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "train_model.py"),
                "--real", str(real_csv),
                "--fake", str(fake_csv),
                "--outdir", str(outdir),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0, f"Training failed: {result.stderr}"
        assert (outdir / "pipeline.joblib").exists()
        assert (outdir / "metrics.json").exists()


class TestDetectFakeNewsScriptReal:
    @pytest.fixture
    def trained_model(self, tmp_path):
        real_csv = tmp_path / "real.csv"
        fake_csv = tmp_path / "fake.csv"
        outdir = tmp_path / "model_outputs"

        real_data = pd.DataFrame({
            "title": [
                "Scientists Publish Research Results",
                "Government Announces New Policy",
                "Local Council Approves Budget",
                "Research Team Makes Discovery",
                "Official Statement Released",
                "Study Shows Health Benefits",
                "Court Rules on Case",
                "Economic Report Published",
                "Weather Forecast Updated",
                "Sports Team Wins Championship",
            ],
            "text": [
                "Scientists at a major university have published their research findings in a peer-reviewed journal.",
                "The government today announced a new policy initiative affecting multiple sectors.",
                "The local city council has approved the annual budget after deliberation.",
                "A research team has made an important discovery in the field of medicine.",
                "An official statement was released regarding the recent developments.",
                "A new study shows significant health benefits of regular exercise.",
                "The court has ruled on the case after reviewing all evidence presented.",
                "The economic report for this quarter has been published by analysts.",
                "The weather forecast has been updated for the coming week.",
                "The local sports team has won the championship after a successful season.",
            ],
            "subject": ["science"] * 10,
            "date": ["2024-01-01"] * 10,
        })
        real_data.to_csv(real_csv, index=False)

        fake_data = pd.DataFrame({
            "title": [
                "SHOCKING: Government Secret REVEALED!",
                "Doctors HATE This One WEIRD Trick!",
                "You Won't BELIEVE What Happened!",
                "MIRACLE Cure They're HIDING!",
                "CELEBRITY Scandal EXPOSED!",
                "BREAKING: Alien Contact CONFIRMED!",
                "Scientists SHOCKED By Discovery!",
                "The TRUTH They Don't Want You To Know!",
                "MIND-BLOWING Secret FINALLY Revealed!",
                "URGENT: This Will CHANGE Everything!",
            ],
            "text": [
                "The government has been hiding this secret for years! You won't believe what we found!",
                "Doctors are furious about this one simple trick that cures everything naturally!",
                "Shocking news that will completely blow your mind! Click here immediately!",
                "Big Pharma doesn't want you to know about this miracle cure! Read now!",
                "Celebrity scandal exposed! Hollywood insiders reveal shocking truth!",
                "Aliens have made contact and the government is hiding it! See the evidence!",
                "Scientists are shocked by this discovery that changes everything we know!",
                "The truth they've been hiding from you is finally revealed! Don't miss this!",
                "This mind-blowing secret has been kept from the public for decades!",
                "Urgent news that will change everything you thought you knew! Act now!",
            ],
            "subject": ["fake"] * 10,
            "date": ["2024-01-01"] * 10,
        })
        fake_data.to_csv(fake_csv, index=False)

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "train_model.py"),
                "--real", str(real_csv),
                "--fake", str(fake_csv),
                "--outdir", str(outdir),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert (outdir / "pipeline.joblib").exists(), "Training failed to produce pipeline.joblib"
        return outdir

    def test_detect_fake_news_predicts_real_news_correctly(self, trained_model):
        real_news_texts = [
            "Scientists at the university published their research findings today.",
            "The government announced a new economic policy this morning.",
            "Local officials confirmed the election results yesterday.",
        ]

        for text in real_news_texts:
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "src" / "detect_fake_news.py"),
                    "--pipeline", str(trained_model / "pipeline.joblib"),
                    "--text", text,
                    "--threshold", "0.5",
                ],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )

            assert result.returncode == 0
            output = result.stdout.strip()
            assert "Label: REAL" in output, f"Expected REAL for '{text}', got: {output}"

    def test_detect_fake_news_predicts_fake_news_correctly(self, trained_model):
        fake_news_texts = [
            "SHOCKING: Government hiding the truth about aliens! You won't believe this! Click here now!",
            "MIRACLE cure they don't want you to know! Doctors HATE this! Must read immediately!",
            "BREAKING: Celebrity scandal EXPOSED! Hollywood insiders reveal SHOCKING truth! Share now!",
        ]

        for text in fake_news_texts:
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "src" / "detect_fake_news.py"),
                    "--pipeline", str(trained_model / "pipeline.joblib"),
                    "--text", text,
                    "--threshold", "0.3",
                ],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )

            assert result.returncode == 0
            output = result.stdout.strip()
            assert "Label: FAKE" in output, f"Expected FAKE for '{text}', got: {output}"

    def test_detect_fake_news_with_separate_model_and_vectorizer(self, trained_model):
        text = "Scientists published new research in a peer-reviewed journal."

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "detect_fake_news.py"),
                "--model", str(trained_model / "model.joblib"),
                "--vectorizer", str(trained_model / "vectorizer.joblib"),
                "--text", text,
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0
        output = result.stdout.strip()
        assert "Label:" in output
        assert "Fake probability:" in output

    def test_detect_fake_news_custom_threshold(self, trained_model):
        text = "Some news article with uncertain content"

        result_low = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "detect_fake_news.py"),
                "--pipeline", str(trained_model / "pipeline.joblib"),
                "--text", text,
                "--threshold", "0.1",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        result_high = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "detect_fake_news.py"),
                "--pipeline", str(trained_model / "pipeline.joblib"),
                "--text", text,
                "--threshold", "0.9",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert result_low.returncode == 0
        assert result_high.returncode == 0
        assert "Threshold: 0.10" in result_low.stdout
        assert "Threshold: 0.90" in result_high.stdout

    def test_detect_fake_news_output_format(self, trained_model):
        text = "Test news article"

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "detect_fake_news.py"),
                "--pipeline", str(trained_model / "pipeline.joblib"),
                "--text", text,
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        output = result.stdout.strip()
        assert "Label:" in output
        assert "Fake probability:" in output
        assert "Threshold:" in output
        assert "|" in output


class TestEndToEndScriptIntegration:
    def test_full_pipeline_from_training_to_prediction(self, tmp_path):
        real_csv = tmp_path / "real_news.csv"
        fake_csv = tmp_path / "fake_news.csv"
        outdir = tmp_path / "trained_model"

        real_titles = [
            "Research Team Publishes Study",
            "Government Releases Report",
            "Court Announces Decision",
            "University Opens New Facility",
            "Company Announces Merger",
            "Official Statement Released",
            "Study Shows Health Benefits",
            "Economic Growth Reported",
            "New Policy Implemented",
            "Scientific Discovery Made",
        ]
        real_texts = [
            "A research team has published a comprehensive study in a scientific journal.",
            "The government has released its annual report on economic performance.",
            "The court announced its decision after reviewing all submitted evidence.",
            "The university has opened a new research facility on campus.",
            "Two major companies announced a merger agreement today.",
            "An official statement was released regarding the recent developments.",
            "A new study shows significant health benefits of regular exercise.",
            "Economic growth was reported in the latest quarterly analysis.",
            "A new policy has been implemented following public consultation.",
            "Scientists have made an important discovery in their research.",
        ]

        real_data = pd.DataFrame({
            "title": real_titles,
            "text": real_texts,
            "subject": ["news"] * 10,
            "date": ["2024-01-01"] * 10,
        })
        real_data.to_csv(real_csv, index=False)

        fake_titles = [
            "SHOCKING Discovery Will BLOW YOUR MIND!",
            "Doctors HATE This Simple Trick!",
            "You Won't BELIEVE The Truth!",
            "MIRACLE Secret FINALLY Revealed!",
            "URGENT: Government HIDING This!",
            "SHOCKING Revelation EXPOSED!",
            "Scientists BAFFLED By Discovery!",
            "The TRUTH They Hide From You!",
            "MIND-BLOWING Secret Revealed!",
            "BREAKING: This Changes Everything!",
        ]
        fake_texts = [
            "This shocking discovery will completely blow your mind! Scientists are baffled!",
            "Doctors absolutely hate this one simple trick! Click to find out more!",
            "You simply won't believe the truth we've uncovered! Must read!",
            "The miracle secret is finally revealed after being hidden for years!",
            "Urgent news: the government is hiding this from you! Share immediately!",
            "Shocking revelation exposed! You won't believe what we found!",
            "Scientists are baffled by this discovery! Click here for details!",
            "The truth they hide from you is finally revealed! Read now!",
            "Mind-blowing secret revealed! Don't miss this shocking information!",
            "Breaking news that will change everything you know! Act now!",
        ]

        fake_data = pd.DataFrame({
            "title": fake_titles,
            "text": fake_texts,
            "subject": ["fake"] * 10,
            "date": ["2024-01-01"] * 10,
        })
        fake_data.to_csv(fake_csv, index=False)

        train_result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "train_model.py"),
                "--real", str(real_csv),
                "--fake", str(fake_csv),
                "--outdir", str(outdir),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert train_result.returncode == 0, f"Training failed: {train_result.stderr}"
        assert (outdir / "pipeline.joblib").exists()
        assert (outdir / "metrics.json").exists()

        real_news = "The research team published their findings in a peer-reviewed scientific journal."
        predict_real = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "detect_fake_news.py"),
                "--pipeline", str(outdir / "pipeline.joblib"),
                "--text", real_news,
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert predict_real.returncode == 0
        assert "Label: REAL" in predict_real.stdout, f"Expected REAL, got: {predict_real.stdout}"

        fake_news = "SHOCKING truth they don't want you to know! You won't believe this!"
        predict_fake = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "detect_fake_news.py"),
                "--pipeline", str(outdir / "pipeline.joblib"),
                "--text", fake_news,
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert predict_fake.returncode == 0
        assert "Label: FAKE" in predict_fake.stdout, f"Expected FAKE, got: {predict_fake.stdout}"

    def test_pipeline_with_separate_artifacts(self, tmp_path):
        real_csv = tmp_path / "real.csv"
        fake_csv = tmp_path / "fake.csv"
        outdir = tmp_path / "model"

        real_data = pd.DataFrame({
            "title": ["Official Announcement"] * 5,
            "text": [
                "The official announcement was made today regarding the new policy.",
                "Government officials confirmed the details in a press conference.",
                "The report has been verified by multiple independent sources.",
                "Experts have reviewed and validated the research findings.",
                "The data was collected through rigorous scientific methods.",
            ],
            "subject": ["news"] * 5,
            "date": ["2024-01-01"] * 5,
        })
        real_data.to_csv(real_csv, index=False)

        fake_data = pd.DataFrame({
            "title": ["SHOCKING REVELATION!"] * 5,
            "text": [
                "SHOCKING revelation that will blow your mind! Click now!",
                "You won't believe what we discovered! Must share immediately!",
                "The truth they're hiding from you! Read before it's deleted!",
                "Scientists are BAFFLED by this discovery! You won't believe it!",
                "This MIRACLE solution they don't want you to know about!",
            ],
            "subject": ["fake"] * 5,
            "date": ["2024-01-01"] * 5,
        })
        fake_data.to_csv(fake_csv, index=False)

        train_result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "train_model.py"),
                "--real", str(real_csv),
                "--fake", str(fake_csv),
                "--outdir", str(outdir),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert train_result.returncode == 0

        real_text = "The official statement was confirmed by multiple sources."
        predict_result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "detect_fake_news.py"),
                "--model", str(outdir / "model.joblib"),
                "--vectorizer", str(outdir / "vectorizer.joblib"),
                "--text", real_text,
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert predict_result.returncode == 0
        assert "Label:" in predict_result.stdout

    def test_metrics_accuracy_after_training(self, tmp_path):
        real_csv = tmp_path / "real.csv"
        fake_csv = tmp_path / "fake.csv"
        outdir = tmp_path / "outputs"

        real_data = pd.DataFrame({
            "title": [f"Real News Title {i}" for i in range(20)],
            "text": [f"This is legitimate news article number {i} with factual content." for i in range(20)],
            "subject": ["news"] * 20,
            "date": ["2024-01-01"] * 20,
        })
        real_data.to_csv(real_csv, index=False)

        fake_data = pd.DataFrame({
            "title": [f"SHOCKING Title {i}!" for i in range(20)],
            "text": [f"SHOCKING content you won't believe! Click here now! Number {i}" for i in range(20)],
            "subject": ["fake"] * 20,
            "date": ["2024-01-01"] * 20,
        })
        fake_data.to_csv(fake_csv, index=False)

        train_result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "train_model.py"),
                "--real", str(real_csv),
                "--fake", str(fake_csv),
                "--outdir", str(outdir),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert train_result.returncode == 0

        with open(outdir / "metrics.json") as f:
            metrics = json.load(f)

        assert "accuracy" in metrics
        assert "roc_auc" in metrics
        assert "report" in metrics
        assert "REAL" in metrics["report"]
        assert "FAKE" in metrics["report"]
        assert metrics["accuracy"] >= 0.5

    def test_model_persistence_and_reloading(self, tmp_path):
        real_csv = tmp_path / "real.csv"
        fake_csv = tmp_path / "fake.csv"
        outdir = tmp_path / "model"

        real_data = pd.DataFrame({
            "title": [
                "Legitimate News One",
                "Legitimate News Two",
                "Legitimate News Three",
                "Legitimate News Four",
                "Legitimate News Five",
            ],
            "text": [
                "This is a legitimate news article with factual information about science.",
                "Government officials confirmed the new policy in a press conference today.",
                "Research findings were published in a peer-reviewed scientific journal.",
                "The official report was released after thorough investigation.",
                "Experts validated the data through rigorous scientific analysis.",
            ],
            "subject": ["news"] * 5,
            "date": ["2024-01-01"] * 5,
        })
        real_data.to_csv(real_csv, index=False)

        fake_data = pd.DataFrame({
            "title": [
                "SHOCKING Clickbait One!",
                "SHOCKING Clickbait Two!",
                "SHOCKING Clickbait Three!",
                "SHOCKING Clickbait Four!",
                "SHOCKING Clickbait Five!",
            ],
            "text": [
                "You won't believe this shocking clickbait content! Click now!",
                "Doctors HATE this one weird trick! You must read immediately!",
                "SHOCKING truth they don't want you to know! Share this!",
                "MIRACLE discovery will BLOW YOUR MIND! Don't miss this!",
                "The government is HIDING this from you! Read before deleted!",
            ],
            "subject": ["fake"] * 5,
            "date": ["2024-01-01"] * 5,
        })
        fake_data.to_csv(fake_csv, index=False)

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "train_model.py"),
                "--real", str(real_csv),
                "--fake", str(fake_csv),
                "--outdir", str(outdir),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0, f"Training failed: {result.stderr}"

        pipeline = joblib.load(outdir / "pipeline.joblib")
        assert hasattr(pipeline, "predict")
        assert hasattr(pipeline, "predict_proba")

        test_text = "Test article content"
        prob = pipeline.predict_proba([test_text])[0, 1]
        assert 0.0 <= prob <= 1.0
