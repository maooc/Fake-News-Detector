"""
Unit tests for text_clean module.
"""
import sys
sys.path.insert(0, "src")

import pytest
from text_clean import clean_text, clean_many


class TestCleanText:
    """Tests for clean_text function."""

    def test_empty_input(self):
        """Test handling of empty or None input."""
        assert clean_text(None) == ""
        assert clean_text("") == ""
        assert clean_text("   ") == ""

    def test_non_string_input(self):
        """Test handling of non-string input types."""
        assert clean_text(123) == ""
        assert clean_text(3.14) == ""
        assert clean_text([]) == ""
        assert clean_text({}) == ""
        assert clean_text(True) == ""

    def test_lowercase_conversion(self):
        """Test lowercase conversion."""
        assert clean_text("HELLO WORLD", lowercase=True) == "hello world"
        assert clean_text("HELLO WORLD", lowercase=False) == "HELLO WORLD"

    def test_url_removal(self):
        """Test URL removal functionality."""
        # URLs should be removed
        assert clean_text("Check https://example.com", remove_urls=True) == "check"
        assert clean_text("Visit http://test.com/page", remove_urls=True) == "visit"
        # URLs should remain if flag is False
        assert "https://example.com" in clean_text("Check https://example.com", remove_urls=False)

    def test_email_removal(self):
        """Test email removal functionality."""
        assert clean_text("Contact test@email.com", remove_emails=True) == "contact"
        assert clean_text("Email me@example.org for info", remove_emails=True) == "email for info"
        # Emails should remain if flag is False
        assert "test@email.com" in clean_text("Contact test@email.com", remove_emails=False)

    def test_non_ascii_removal(self):
        """Test non-ASCII character removal."""
        # Non-ASCII should be removed (replaced with space then collapsed)
        result = clean_text("café español", remove_non_ascii=True)
        assert result in ["caf espa ol", "caf espaol"]  # May have extra space from removal
        assert clean_text("日本語 中文", remove_non_ascii=True) == ""
        # Non-ASCII should remain if flag is False
        result = clean_text("café español", remove_non_ascii=False)
        assert "é" in result or "ñ" in result

    def test_whitespace_collapsing(self):
        """Test whitespace collapsing."""
        # Multiple spaces should collapse to single
        assert clean_text("   hello   world   ", collapse_whitespace=True) == "hello world"
        assert clean_text("hello\n\nworld\t\t!", collapse_whitespace=True) == "hello world !"
        # Whitespace preserved if flag is False (may have leading/trailing)
        result = clean_text("   hello   ", collapse_whitespace=False)
        assert len(result) > len("hello")

    def test_combined_cleaning(self):
        """Test combined cleaning with all flags."""
        dirty_text = """
        HELLO https://spam.com contact@spam.org café  
        multiple   spaces   
        """
        result = clean_text(dirty_text)
        # After URL and email removal and cleaning, we should get hello followed by caf-related content
        assert "hello" in result
        assert "caf" in result
        assert "multiple spaces" in result
        # URLs and emails should be removed
        assert "https://spam.com" not in result
        assert "contact@spam.org" not in result

    def test_partial_flags(self):
        """Test cleaning with partial flags."""
        text = "HELLO https://test.com"
        # Only lowercase, keep URL
        result = clean_text(text, remove_urls=False, remove_emails=False, 
                          remove_non_ascii=False, collapse_whitespace=True)
        assert result == "hello https://test.com"

    def test_punctuation_handling(self):
        """Test that punctuation is preserved (as per current implementation)."""
        # Current implementation doesn't remove punctuation
        text = "Hello, world! How are you?"
        result = clean_text(text, lowercase=False)
        assert "Hello, world!" in result or "Hello , world !" in result or result == "Hello, world! How are you?"


class TestCleanMany:
    """Tests for clean_many function."""

    def test_empty_list(self):
        """Test cleaning empty list."""
        assert clean_many([]) == []

    def test_list_with_none(self):
        """Test cleaning list with None values."""
        result = clean_many(["Hello", None, "World"])
        assert result == ["hello", "", "world"]

    def test_mixed_content(self):
        """Test cleaning list with mixed content."""
        texts = [
            "HELLO https://test.com",
            "contact@email.com",
            "café español",
            None,
            "   multiple   spaces   "
        ]
        results = clean_many(texts)
        assert len(results) == 5
        assert results[0] == "hello"
        assert results[1] == ""
        # May have extra space from non-ASCII removal
        assert results[2] in ["caf espa ol", "caf espaol"]
        assert results[3] == ""
        assert results[4] == "multiple spaces"

    def test_custom_kwargs(self):
        """Test that kwargs are passed to clean_text."""
        texts = ["HELLO WORLD", "https://keep-url.com"]
        results = clean_many(texts, lowercase=False, remove_urls=False)
        assert results[0] == "HELLO WORLD"
        assert "https://keep-url.com" in results[1]

    def test_sequence_types(self):
        """Test handling different sequence types."""
        from collections.abc import Sequence
        
        # List
        assert clean_many(["a", "b", "c"]) == ["a", "b", "c"]
        # Tuple
        assert clean_many(("x", "y", "z")) == ["x", "y", "z"]

    def test_large_input(self):
        """Test with larger input size."""
        texts = ["Test text " + str(i) for i in range(100)]
        results = clean_many(texts)
        assert len(results) == 100
        assert all(isinstance(r, str) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
