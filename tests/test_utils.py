import json
import pytest
from pathlib import Path
from utils import ensure_outdir, save_json, load_json


class TestEnsureOutdir:
    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "new_directory"
        assert not new_dir.exists()
        result = ensure_outdir(new_dir)
        assert new_dir.exists()
        assert result == new_dir.resolve()

    def test_existing_directory(self, tmp_path):
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        result = ensure_outdir(existing_dir)
        assert existing_dir.exists()
        assert result == existing_dir.resolve()

    def test_nested_directory_creation(self, tmp_path):
        nested = tmp_path / "level1" / "level2" / "level3"
        result = ensure_outdir(nested)
        assert nested.exists()
        assert result == nested.resolve()

    def test_with_string_path(self, tmp_path):
        dir_path = str(tmp_path / "string_path_dir")
        result = ensure_outdir(dir_path)
        assert Path(dir_path).exists()


class TestSaveJson:
    def test_save_simple_dict(self, temp_json_file):
        data = {"key": "value", "number": 42}
        result = save_json(data, temp_json_file)
        assert result == temp_json_file.resolve()
        assert temp_json_file.exists()

    def test_save_nested_dict(self, temp_json_file):
        data = {
            "outer": {
                "inner": {"deep": "value"}
            },
            "list": [1, 2, 3]
        }
        save_json(data, temp_json_file)
        loaded = json.loads(temp_json_file.read_text())
        assert loaded["outer"]["inner"]["deep"] == "value"

    def test_save_list(self, temp_json_file):
        data = [1, 2, 3, "four", {"five": 5}]
        save_json(data, temp_json_file)
        loaded = json.loads(temp_json_file.read_text())
        assert loaded == data

    def test_indent_parameter(self, temp_json_file):
        data = {"key": "value"}
        save_json(data, temp_json_file, indent=4)
        content = temp_json_file.read_text()
        assert "    " in content

    def test_creates_parent_directory(self, tmp_path):
        nested_file = tmp_path / "subdir" / "nested" / "data.json"
        save_json({"test": "data"}, nested_file)
        assert nested_file.exists()

    def test_unicode_content(self, temp_json_file):
        data = {"chinese": "中文测试", "emoji": "😀"}
        save_json(data, temp_json_file)
        loaded = load_json(temp_json_file)
        assert loaded["chinese"] == "中文测试"
        assert loaded["emoji"] == "😀"

    def test_with_path_object(self, temp_json_file):
        data = {"path": "object"}
        save_json(data, temp_json_file)
        assert temp_json_file.exists()

    def test_with_string_path(self, tmp_path):
        file_path = str(tmp_path / "string_path.json")
        save_json({"test": "string"}, file_path)
        assert Path(file_path).exists()


class TestLoadJson:
    def test_load_simple_dict(self, temp_json_file):
        data = {"key": "value"}
        save_json(data, temp_json_file)
        loaded = load_json(temp_json_file)
        assert loaded == data

    def test_load_nested_structure(self, temp_json_file):
        data = {
            "metrics": {
                "accuracy": 0.95,
                "f1": 0.93
            },
            "labels": ["REAL", "FAKE"]
        }
        save_json(data, temp_json_file)
        loaded = load_json(temp_json_file)
        assert loaded["metrics"]["accuracy"] == 0.95
        assert loaded["labels"] == ["REAL", "FAKE"]

    def test_load_list(self, temp_json_file):
        data = [1, 2, 3]
        save_json(data, temp_json_file)
        loaded = load_json(temp_json_file)
        assert loaded == data

    def test_load_nonexistent_file(self, tmp_path):
        nonexistent = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            load_json(nonexistent)

    def test_load_invalid_json(self, tmp_path):
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{ invalid json }")
        with pytest.raises(json.JSONDecodeError):
            load_json(invalid_file)

    def test_load_with_string_path(self, temp_json_file):
        save_json({"test": "data"}, temp_json_file)
        loaded = load_json(str(temp_json_file))
        assert loaded["test"] == "data"


class TestJsonRoundTrip:
    def test_roundtrip_preserves_data(self, temp_json_file):
        original = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": 2}
        }
        save_json(original, temp_json_file)
        loaded = load_json(temp_json_file)
        assert loaded == original

    def test_roundtrip_with_special_characters(self, temp_json_file):
        original = {
            "newline": "line1\nline2",
            "tab": "col1\tcol2",
            "quote": 'He said "hello"'
        }
        save_json(original, temp_json_file)
        loaded = load_json(temp_json_file)
        assert loaded == original
