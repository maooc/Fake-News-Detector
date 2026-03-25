"""
Unit tests for utils module.
Tests file I/O operations and JSON save/load functionality.
"""

import json
from pathlib import Path

import pytest

from utils import ensure_outdir, save_json, load_json


class TestEnsureOutdir:
    """Tests for ensure_outdir function."""

    def test_create_new_directory(self, tmp_path):
        """Test creating a new directory."""
        new_dir = tmp_path / "new_directory"
        result = ensure_outdir(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir.resolve()

    def test_create_nested_directories(self, tmp_path):
        """Test creating nested directories."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        result = ensure_outdir(nested_dir)
        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_existing_directory(self, tmp_path):
        """Test ensuring an already existing directory."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        result = ensure_outdir(existing_dir)
        assert existing_dir.exists()
        assert result == existing_dir.resolve()

    def test_with_string_path(self, tmp_path):
        """Test with string path instead of Path object."""
        str_path = str(tmp_path / "string_path_dir")
        result = ensure_outdir(str_path)
        assert Path(str_path).exists()
        assert isinstance(result, Path)


class TestSaveJson:
    """Tests for save_json function."""

    def test_save_simple_dict(self, tmp_path):
        """Test saving a simple dictionary."""
        data = {"key": "value", "number": 42}
        file_path = tmp_path / "test.json"
        result = save_json(data, file_path)
        
        assert file_path.exists()
        assert result == file_path.resolve()
        
        with open(file_path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_nested_structure(self, tmp_path):
        """Test saving nested dictionary and list structures."""
        data = {
            "level1": {
                "level2": {
                    "list": [1, 2, 3],
                    "string": "test"
                }
            },
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.92
            }
        }
        file_path = tmp_path / "nested.json"
        save_json(data, file_path)
        
        with open(file_path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_creates_parent_directories(self, tmp_path):
        """Test that save_json creates parent directories."""
        data = {"test": "data"}
        file_path = tmp_path / "subdir1" / "subdir2" / "file.json"
        save_json(data, file_path)
        
        assert file_path.exists()

    def test_save_with_string_path(self, tmp_path):
        """Test saving with string path."""
        data = {"key": "value"}
        str_path = str(tmp_path / "string.json")
        result = save_json(data, str_path)
        
        assert Path(str_path).exists()
        assert isinstance(result, Path)

    def test_save_with_custom_indent(self, tmp_path):
        """Test saving with custom indentation."""
        data = {"key": "value"}
        file_path = tmp_path / "indent.json"
        save_json(data, file_path, indent=4)
        
        with open(file_path) as f:
            content = f.read()
        # Check that indentation is 4 spaces
        assert "    " in content

    def test_save_list(self, tmp_path):
        """Test saving a list instead of dict."""
        data = [1, 2, 3, "test", {"nested": True}]
        file_path = tmp_path / "list.json"
        save_json(data, file_path)
        
        with open(file_path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_with_special_characters(self, tmp_path):
        """Test saving strings with special characters."""
        data = {"text": "Hello 世界", "emoji": "🎉", "symbols": "<>&"}
        file_path = tmp_path / "special.json"
        save_json(data, file_path)
        
        with open(file_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["text"] == "Hello 世界"
        assert loaded["emoji"] == "🎉"


class TestLoadJson:
    """Tests for load_json function."""

    def test_load_simple_dict(self, tmp_path):
        """Test loading a simple JSON file."""
        data = {"key": "value", "number": 42}
        file_path = tmp_path / "test.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
        
        loaded = load_json(file_path)
        assert loaded == data

    def test_load_nested_structure(self, tmp_path):
        """Test loading nested JSON structure."""
        data = {
            "metrics": {
                "accuracy": 0.95,
                "report": {
                    "precision": 0.92,
                    "recall": 0.88
                }
            }
        }
        file_path = tmp_path / "nested.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
        
        loaded = load_json(file_path)
        assert loaded["metrics"]["accuracy"] == 0.95
        assert loaded["metrics"]["report"]["precision"] == 0.92

    def test_load_with_string_path(self, tmp_path):
        """Test loading with string path."""
        data = {"key": "value"}
        str_path = str(tmp_path / "string.json")
        with open(str_path, "w") as f:
            json.dump(data, f)
        
        loaded = load_json(str_path)
        assert loaded == data

    def test_load_list(self, tmp_path):
        """Test loading a JSON list."""
        data = [1, 2, 3, {"test": True}]
        file_path = tmp_path / "list.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
        
        loaded = load_json(file_path)
        assert loaded == data

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading a non-existent file raises error."""
        nonexistent = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            load_json(nonexistent)

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON raises error."""
        file_path = tmp_path / "invalid.json"
        with open(file_path, "w") as f:
            f.write("not valid json {")
        
        with pytest.raises(json.JSONDecodeError):
            load_json(file_path)


class TestJsonRoundTrip:
    """Tests for save/load round-trip operations."""

    def test_round_trip_simple(self, tmp_path):
        """Test save and load round-trip with simple data."""
        original = {"key": "value", "number": 42}
        file_path = tmp_path / "roundtrip.json"
        
        save_json(original, file_path)
        loaded = load_json(file_path)
        
        assert loaded == original

    def test_round_trip_complex(self, tmp_path):
        """Test round-trip with complex nested data."""
        original = {
            "model": "FakeNewsDetector",
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.92,
                "recall": 0.88,
                "f1": 0.90
            },
            "config": {
                "vectorizer": "TF-IDF",
                "classifier": "RandomForest",
                "params": {
                    "n_estimators": 100,
                    "max_depth": None
                }
            },
            "samples": [
                {"id": 1, "label": "REAL"},
                {"id": 2, "label": "FAKE"}
            ]
        }
        file_path = tmp_path / "complex.json"
        
        save_json(original, file_path)
        loaded = load_json(file_path)
        
        assert loaded == original

    def test_round_trip_unicode(self, tmp_path):
        """Test round-trip with unicode characters."""
        original = {
            "chinese": "你好世界",
            "japanese": "こんにちは",
            "emoji": "🎉🎊",
            "math": "∑∏∫"
        }
        file_path = tmp_path / "unicode.json"
        
        save_json(original, file_path)
        loaded = load_json(file_path)
        
        assert loaded == original


class TestEdgeCases:
    """Edge case tests for utils module."""

    def test_save_empty_dict(self, tmp_path):
        """Test saving empty dictionary."""
        file_path = tmp_path / "empty.json"
        save_json({}, file_path)
        
        loaded = load_json(file_path)
        assert loaded == {}

    def test_save_empty_list(self, tmp_path):
        """Test saving empty list."""
        file_path = tmp_path / "empty_list.json"
        save_json([], file_path)
        
        loaded = load_json(file_path)
        assert loaded == []

    def test_save_none_values(self, tmp_path):
        """Test saving None values."""
        data = {"null_value": None, "list": [1, None, 3]}
        file_path = tmp_path / "nulls.json"
        save_json(data, file_path)
        
        loaded = load_json(file_path)
        assert loaded["null_value"] is None
        assert loaded["list"][1] is None

    def test_save_boolean_values(self, tmp_path):
        """Test saving boolean values."""
        data = {"true_val": True, "false_val": False}
        file_path = tmp_path / "bools.json"
        save_json(data, file_path)
        
        loaded = load_json(file_path)
        assert loaded["true_val"] is True
        assert loaded["false_val"] is False

    def test_save_numeric_types(self, tmp_path):
        """Test saving various numeric types."""
        data = {
            "int": 42,
            "float": 3.14159,
            "negative": -100,
            "scientific": 1.5e-10,
            "zero": 0
        }
        file_path = tmp_path / "numbers.json"
        save_json(data, file_path)
        
        loaded = load_json(file_path)
        assert loaded["int"] == 42
        assert abs(loaded["float"] - 3.14159) < 1e-10
        assert loaded["negative"] == -100
