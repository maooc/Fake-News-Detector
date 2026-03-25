"""
Unit tests for text_clean module.
Tests various text cleaning scenarios including punctuation, URLs, ASCII, empty strings, etc.
"""

import pytest

from text_clean import clean_text, clean_many


class TestCleanText:
    """Tests for clean_text function."""

    def test_basic_lowercase(self):
        """Test basic lowercase conversion."""
        assert clean_text("HELLO WORLD") == "hello world"
        assert clean_text("Hello World") == "hello world"

    def test_url_removal(self):
        """Test URL removal from text."""
        text = "Check out https://example.com for more info"
        result = clean_text(text)
        assert "https://example.com" not in result
        assert "check out" in result
        assert "for more info" in result

    def test_http_url_removal(self):
        """Test HTTP URL removal."""
        text = "Visit http://test.org/page for details"
        result = clean_text(text)
        assert "http://test.org/page" not in result

    def test_email_removal(self):
        """Test email address removal."""
        text = "Contact us at support@example.com for help"
        result = clean_text(text)
        assert "support@example.com" not in result
        assert "contact us at" in result
        assert "for help" in result

    def test_non_ascii_removal(self):
        """Test non-ASCII character removal."""
        text = "Hello 世界 café naïve"
        result = clean_text(text)
        assert "世界" not in result
        # é and ï are non-ASCII and get removed, leaving "cafe" and "naive"
        assert "hello" in result
        assert "cafe" in result or "caf" in result
        assert "naive" in result or "na" in result

    def test_punctuation_handling(self):
        """Test handling of punctuation (should be preserved in lowercase)."""
        text = "Hello, World! How are you?"
        result = clean_text(text)
        assert result == "hello, world! how are you?"

    def test_multiple_whitespace_collapse(self):
        """Test collapsing multiple whitespaces into single space."""
        text = "Hello    world   \t\n  test"
        result = clean_text(text)
        assert result == "hello world test"

    def test_leading_trailing_whitespace_removal(self):
        """Test removal of leading and trailing whitespace."""
        text = "   hello world   "
        result = clean_text(text)
        assert result == "hello world"
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_empty_string(self):
        """Test handling of empty string."""
        assert clean_text("") == ""

    def test_none_input(self):
        """Test handling of None input."""
        assert clean_text(None) == ""

    def test_non_string_input(self):
        """Test handling of non-string input."""
        assert clean_text(123) == ""
        assert clean_text(["list"]) == ""
        assert clean_text({"key": "value"}) == ""

    def test_complex_mixed_content(self):
        """Test cleaning complex text with URLs, emails, and non-ASCII."""
        text = "SHOCKING!!! Visit https://fake-news.com or email info@lies.com!!! 震惊世界"
        result = clean_text(text)
        assert "https://fake-news.com" not in result
        assert "info@lies.com" not in result
        assert "震惊世界" not in result
        assert "shocking" in result

    def test_disable_lowercase(self):
        """Test disabling lowercase conversion."""
        text = "HELLO World"
        result = clean_text(text, lowercase=False)
        assert "HELLO" in result
        assert "World" in result

    def test_disable_url_removal(self):
        """Test disabling URL removal."""
        text = "Visit https://example.com"
        result = clean_text(text, remove_urls=False)
        assert "https://example.com" in result

    def test_disable_email_removal(self):
        """Test disabling email removal."""
        text = "Email test@example.com"
        result = clean_text(text, remove_emails=False)
        assert "test@example.com" in result

    def test_disable_non_ascii_removal(self):
        """Test disabling non-ASCII removal."""
        text = "Hello 世界"
        result = clean_text(text, remove_non_ascii=False)
        assert "世界" in result

    def test_disable_whitespace_collapse(self):
        """Test disabling whitespace collapse."""
        text = "hello    world"
        result = clean_text(text, collapse_whitespace=False)
        assert "    " in result


class TestCleanMany:
    """Tests for clean_many function."""

    def test_clean_list_of_texts(self):
        """Test cleaning a list of texts."""
        texts = [
            "Hello World",
            "Visit https://example.com",
            "Email test@test.com",
        ]
        results = clean_many(texts)
        assert len(results) == 3
        assert results[0] == "hello world"
        assert "https://example.com" not in results[1]
        assert "test@test.com" not in results[2]

    def test_clean_with_none_in_list(self):
        """Test cleaning list containing None values."""
        texts = ["Hello World", None, "Test"]
        results = clean_many(texts)
        assert results == ["hello world", "", "test"]

    def test_empty_list(self):
        """Test cleaning empty list."""
        assert clean_many([]) == []

    def test_clean_many_preserves_order(self):
        """Test that clean_many preserves input order."""
        texts = ["First", "Second", "Third"]
        results = clean_many(texts)
        assert results == ["first", "second", "third"]

    def test_clean_many_with_custom_options(self):
        """Test clean_many with custom cleaning options."""
        texts = ["HELLO https://test.com"]
        results = clean_many(texts, lowercase=False, remove_urls=False)
        assert results[0] == "HELLO https://test.com"


class TestEdgeCases:
    """Edge case tests for text cleaning."""

    def test_only_urls(self):
        """Test text containing only URLs."""
        text = "https://example.com http://test.org"
        result = clean_text(text)
        assert result == ""

    def test_only_emails(self):
        """Test text containing only emails."""
        text = "test@example.com another@test.org"
        result = clean_text(text)
        assert result == ""

    def test_only_non_ascii(self):
        """Test text containing only non-ASCII."""
        text = "你好世界こんにちは"
        result = clean_text(text)
        assert result == ""

    def test_only_whitespace(self):
        """Test text containing only whitespace."""
        text = "     \t\n   "
        result = clean_text(text)
        assert result == ""

    def test_single_character(self):
        """Test single character input."""
        assert clean_text("A") == "a"
        assert clean_text("!") == "!"

    def test_very_long_text(self):
        """Test handling of very long text."""
        text = "word " * 10000
        result = clean_text(text)
        assert len(result) > 0
        assert "word" in result
