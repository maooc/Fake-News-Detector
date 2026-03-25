import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def mini_real_csv(fixtures_dir):
    return fixtures_dir / "mini_real.csv"


@pytest.fixture
def mini_fake_csv(fixtures_dir):
    return fixtures_dir / "mini_fake.csv"


@pytest.fixture
def mini_real_df(mini_real_csv):
    return pd.read_csv(mini_real_csv)


@pytest.fixture
def mini_fake_df(mini_fake_csv):
    return pd.read_csv(mini_fake_csv)


@pytest.fixture
def sample_texts():
    return [
        "This is a normal news article about politics.",
        "SHOCKING revelation that will BLOW YOUR MIND!!!",
        "Scientists discover new species in the Amazon rainforest.",
        "You won't BELIEVE what happened next! Click here!",
    ]


@pytest.fixture
def dirty_texts():
    return [
        "Check out https://example.com for more info!",
        "Contact us at email@test.com for details.",
        "Text with non-ASCII: café résumé naïve",
        "Multiple   spaces   should   be   collapsed.",
        "UPPERCASE TEXT SHOULD BE LOWERCASED.",
    ]


@pytest.fixture
def temp_output_dir(tmp_path):
    outdir = tmp_path / "outputs"
    outdir.mkdir()
    return outdir


@pytest.fixture
def temp_json_file(tmp_path):
    return tmp_path / "test_data.json"
