"""
Unit tests for utils module.
"""
import sys
sys.path.insert(0, "src")

import pytest
import json
from pathlib import Path
from utils import ensure_outdir, save_json, load_json


class TestEnsureOutdir:
    """Tests for ensure_outdir function."""

    def test_create_new_directory(self, temp_dir):
        """Test creating a new directory."""
        new_dir = temp_dir / "new_output"
        result = ensure_outdir(new_dir)
        assert result.exists()
        assert result.is_dir()
        assert result == new_dir.resolve()

    def test_create_nested_directories(self, temp_dir):
        """Test creating nested directories."""
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        result = ensure_outdir(nested_dir)
        assert result.exists()
        assert result.is_dir()

    def test_existing_directory(self, temp_dir):
        """Test ensure_outdir on already existing directory."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()
        # Create a file inside to verify directory contents preserved
        (existing_dir / "test.txt").write_text("test")
        
        result = ensure_outdir(existing_dir)
        assert result.exists()
        assert (existing_dir / "test.txt").exists()

    def test_string_path_input(self, temp_dir):
        """Test ensure_outdir with string path input."""
        dir_path = str(temp_dir / "string_path")
        result = ensure_outdir(dir_path)
        assert result.exists()
        assert isinstance(result, Path)

    def test_return_resolved_path(self, temp_dir):
        """Test that returned path is resolved (absolute)."""
        # Use relative path from temp_dir
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            result = ensure_outdir("relative_dir")
            assert result.is_absolute()
        finally:
            os.chdir(original_cwd)


class TestSaveJson:
    """Tests for save_json function."""

    def test_save_simple_object(self, temp_dir):
        """Test saving simple JSON-serializable object."""
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        file_path = temp_dir / "output.json"
        
        result = save_json(data, file_path)
        assert result.exists()
        assert result == file_path.resolve()
        
        # Verify content
        with open(result, "r") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_nested_object(self, temp_dir):
        """Test saving nested JSON structure."""
        data = {
            "level1": {
                "level2": {
                    "level3": "deep_value"
                }
            },
            "array": [{"item": 1}, {"item": 2}]
        }
        file_path = temp_dir / "nested.json"
        
        result = save_json(data, file_path)
        assert result.exists()
        
        with open(result, "r") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_with_custom_indent(self, temp_dir):
        """Test saving with custom indentation."""
        data = {"test": "value"}
        file_path = temp_dir / "indented.json"
        
        save_json(data, file_path, indent=4)
        
        # Verify indentation by checking file content length
        content = file_path.read_text()
        assert content.count("    ") > 0  # Should have 4-space indents

    def test_save_creates_parent_dirs(self, temp_dir):
        """Test that save_json creates parent directories if needed."""
        file_path = temp_dir / "deep" / "path" / "to" / "file.json"
        data = {"test": "value"}
        
        result = save_json(data, file_path)
        assert result.exists()
        assert result.parent.exists()

    def test_string_path_for_save(self, temp_dir):
        """Test save_json with string path."""
        data = {"test": "value"}
        file_path = str(temp_dir / "string_path.json")
        
        result = save_json(data, file_path)
        assert result.exists()


class TestLoadJson:
    """Tests for load_json function."""

    def test_load_existing_file(self, temp_dir):
        """Test loading existing JSON file."""
        data = {"key": "value", "number": 42}
        file_path = temp_dir / "test.json"
        
        with open(file_path, "w") as f:
            json.dump(data, f)
        
        loaded = load_json(file_path)
        assert loaded == data

    def test_load_from_string_path(self, temp_dir):
        """Test loading from string path."""
        data = {"test": "value"}
        file_path = temp_dir / "test.json"
        
        with open(file_path, "w") as f:
            json.dump(data, f)
        
        loaded = load_json(str(file_path))
        assert loaded == data

    def test_load_complex_data(self, temp_dir):
        """Test loading complex JSON data."""
        data = {
            "metadata": {
                "version": "1.0",
                "timestamp": "2024-01-01"
            },
            "results": [
                {"id": 1, "score": 0.95},
                {"id": 2, "score": 0.87}
            ]
        }
        file_path = temp_dir / "complex.json"
        
        with open(file_path, "w") as f:
            json.dump(data, f)
        
        loaded = load_json(file_path)
        assert loaded == data
        assert isinstance(loaded["results"], list)
        assert len(loaded["results"]) == 2

    def test_load_nonexistent_file_raises_error(self, temp_dir):
        """Test that loading non-existent file raises appropriate error."""
        file_path = temp_dir / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            load_json(file_path)

    def test_load_invalid_json_raises_error(self, temp_dir):
        """Test that loading invalid JSON raises appropriate error."""
        file_path = temp_dir / "invalid.json"
        file_path.write_text("{not valid json")
        
        with pytest.raises(json.JSONDecodeError):
            load_json(file_path)


class TestRoundTrip:
    """Test save and load round trip."""

    def test_save_and_load_roundtrip(self, temp_dir):
        """Test that data saved can be loaded back unchanged."""
        original_data = {
            "string": "hello world",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"}
        }
        
        file_path = temp_dir / "roundtrip.json"
        save_json(original_data, file_path)
        loaded_data = load_json(file_path)
        
        assert loaded_data == original_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
